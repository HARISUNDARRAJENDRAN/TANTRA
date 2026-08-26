package in.tantra.transceiver.runtime

import android.bluetooth.BluetoothManager
import android.content.Context
import android.net.Uri
import android.os.SystemClock
import in.tantra.transceiver.audio.AdaptiveEndpointDetector
import in.tantra.transceiver.audio.AudioCaptureEngine
import in.tantra.transceiver.audio.PcmAudioPlayer
import in.tantra.transceiver.ml.LoadedModelPack
import in.tantra.transceiver.ml.ModelPackManager
import in.tantra.transceiver.ml.OnnxCtcSttEngine
import in.tantra.transceiver.ml.OnnxTantraTtsEngine
import in.tantra.transceiver.ml.SttEngine
import in.tantra.transceiver.ml.TtsEngine
import in.tantra.transceiver.model.*
import in.tantra.transceiver.protocol.*
import in.tantra.transceiver.transport.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.ByteArrayOutputStream
import java.security.SecureRandom
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

class TantraRuntime(private val context: Context) : AutoCloseable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val mutableState = MutableStateFlow(TantraUiState())
    val state: StateFlow<TantraUiState> = mutableState.asStateFlow()

    private val modelPacks = ModelPackManager(context)
    private val capture = AudioCaptureEngine(context)
    private val detector = AdaptiveEndpointDetector()
    private val player = PcmAudioPlayer(context, scope)
    private val inferenceMutex = Mutex()
    private val audioBuffer = ByteArrayOutputStream()
    private val committer = StablePrefixCommitter()
    private val sessionId = SecureRandom().nextInt().toUInt()
    private val sequence = AtomicInteger(1)
    private val receivedSequences = ConcurrentHashMap.newKeySet<UInt>()
    private val reliableSender = ReliableSender(scope) { transport }

    private var pack: LoadedModelPack? = null
    private var stt: SttEngine? = null
    private var tts: TtsEngine? = null
    private var transport: DuplexTransport? = null
    private var partialJob: Job? = null
    private var captureEnabled = false

    init { useLoopback() }

    suspend fun importModelPack(uri: Uri) {
        update { it.copy(diagnostic = "Verifying model pack…") }
        runCatching { modelPacks.import(uri) }
            .onSuccess { activatePack(it) }
            .onFailure { failure -> update { it.copy(diagnostic = failure.message ?: "Model-pack import failed") } }
    }

    fun activatePack(packId: String) = activatePack(modelPacks.loadActive(packId))

    private fun activatePack(loaded: LoadedModelPack) {
        stt?.close(); tts?.close()
        pack = loaded
        stt = OnnxCtcSttEngine(loaded)
        tts = if (loaded.manifest.tts != null && java.io.File(loaded.directory, "tts.onnx").isFile) {
            OnnxTantraTtsEngine(loaded)
        } else null
        update { it.copy(
            modelPackId = loaded.manifest.packId,
            modelReady = tts != null,
            diagnostic = if (tts == null) "ASR loaded; this pack has no TTS graph" else null,
        ) }
    }

    fun setLanguage(language: Language) = update { it.copy(language = language) }
    fun setAlertMode(enabled: Boolean) = update { it.copy(alertMode = enabled) }

    fun setMode(mode: SessionMode) {
        update { it.copy(mode = mode) }
        if (mode == SessionMode.CONTINUOUS) startCapture() else if (!mutableState.value.isCapturing) stopCaptureAsync()
    }

    fun startPtt() {
        if (!mutableState.value.modelReady) {
            update { it.copy(diagnostic = "Import a compatible ASR + TTS model pack first") }
            return
        }
        captureEnabled = true
        detector.reset(); committer.reset(); audioBuffer.reset()
        update { it.copy(isCapturing = true, localTranscript = "", diagnostic = null) }
        startCapture()
    }

    fun stopPtt() {
        if (!captureEnabled) return
        captureEnabled = false
        scope.launch { finalizeUtterance() }
        update { it.copy(isCapturing = false) }
        if (mutableState.value.mode == SessionMode.PUSH_TO_TALK) stopCaptureAsync()
    }

    private fun startCapture() {
        runCatching {
            capture.start(scope) { frame -> handleAudioFrame(frame) }
            captureEnabled = true
            update { it.copy(isCapturing = true, diagnostic = null) }
        }.onFailure { failure -> update { it.copy(diagnostic = failure.message) } }
    }

    private fun stopCaptureAsync() = scope.launch { capture.stop() }

    private suspend fun handleAudioFrame(frame: ShortArray) {
        val mode = mutableState.value.mode
        if (!captureEnabled && mode != SessionMode.CONTINUOUS) return
        val event = if (mode == SessionMode.PUSH_TO_TALK) AdaptiveEndpointDetector.Event.SPEECH else detector.accept(frame)
        if (mode == SessionMode.CONTINUOUS && event == AdaptiveEndpointDetector.Event.SPEECH_START) {
            audioBuffer.reset(); committer.reset(); captureEnabled = true
            update { it.copy(isCapturing = true) }
        }
        if (captureEnabled) appendPcm(frame)
        if (captureEnabled && audioBuffer.size() >= 12_800 && (partialJob == null || partialJob?.isCompleted == true)) {
            val snapshot = pcmSnapshot()
            partialJob = scope.launch { inferAndCommit(snapshot, final = false) }
        }
        if (mode == SessionMode.CONTINUOUS && event == AdaptiveEndpointDetector.Event.SPEECH_END) {
            captureEnabled = false
            finalizeUtterance()
            update { it.copy(isCapturing = false) }
        }
    }

    private suspend fun inferAndCommit(pcm: ShortArray, final: Boolean) = inferenceMutex.withLock {
        val engine = stt ?: return
        val result = withContext(Dispatchers.Default) { engine.transcribe(pcm, mutableState.value.language) }
        update { state -> state.copy(
            localTranscript = result.text,
            metrics = state.metrics.copy(asrLastMs = result.elapsedMs),
        ) }
        val committed = committer.update(result.text, SystemClock.elapsedRealtime(), final)
        if (committed.isNotBlank()) sendCommitted(committed, final)
    }

    private suspend fun finalizeUtterance() {
        partialJob?.join()
        val snapshot = pcmSnapshot()
        if (snapshot.isNotEmpty()) inferAndCommit(snapshot, final = true)
        audioBuffer.reset(); detector.reset(); committer.reset()
    }

    private suspend fun sendCommitted(text: String, final: Boolean) {
        val activePack = pack ?: return
        val ids = activePack.vocabulary.encode(text)
        val payload = TextDeltaCodec.encode(TextDelta(0u, 0, ids))
        val alert = mutableState.value.alertMode
        val frame = TantraFrame(
            kind = if (alert) FrameKind.ALERT else if (final) FrameKind.FINAL else FrameKind.CLAUSE,
            language = mutableState.value.language,
            priority = if (alert) Priority.ALERT else Priority.NORMAL,
            sessionId = sessionId,
            sequence = sequence.getAndIncrement().toUInt(),
            senderTimestampMs = SystemClock.elapsedRealtime().toULong(),
            payload = payload,
            flags = FrameFlags.TOKEN_PAYLOAD or FrameFlags.FULL_SNAPSHOT or
                (if (final) FrameFlags.FINAL else 0) or FrameFlags.ACK_REQUIRED,
        )
        val encoded = AksharaCodec.encode(frame)
        try {
            reliableSender.send(frame, encoded, if (alert) 8 else 4)
            update { state -> state.copy(metrics = state.metrics.copy(bytesSent = state.metrics.bytesSent + encoded.size)) }
        } catch (failure: Throwable) {
            update { it.copy(diagnostic = "Send failed: ${failure.message}") }
        }
    }

    private suspend fun receive(encoded: ByteArray) {
        val receivedAt = SystemClock.elapsedRealtime()
        val frame = runCatching { AksharaCodec.decode(encoded) }.getOrElse {
            update { state -> state.copy(metrics = state.metrics.copy(droppedFrames = state.metrics.droppedFrames + 1)) }
            return
        }
        update { state -> state.copy(metrics = state.metrics.copy(bytesReceived = state.metrics.bytesReceived + encoded.size)) }
        if (frame.kind == FrameKind.ACK) {
            reliableSender.acknowledge(AckCodec.decode(frame.payload).first)
            return
        }
        if (!receivedSequences.add(frame.sequence)) {
            if (frame.flags and FrameFlags.ACK_REQUIRED != 0) sendAck(frame.sequence, 1)
            return
        }
        if (frame.flags and FrameFlags.ACK_REQUIRED != 0) sendAck(frame.sequence, 0)
        if (frame.kind !in setOf(FrameKind.CLAUSE, FrameKind.FINAL, FrameKind.ALERT)) return
        val activePack = pack ?: return
        val text = if (frame.flags and FrameFlags.TOKEN_PAYLOAD != 0) {
            activePack.vocabulary.decode(TextDeltaCodec.decode(frame.payload).suffixTokens)
        } else frame.payload.decodeToString()
        update { it.copy(remoteTranscript = text, isSpeaking = true) }
        val engine = tts ?: return
        val result = withContext(Dispatchers.Default) { engine.synthesize(text, frame.language) }
        update { state -> state.copy(metrics = state.metrics.copy(ttsLastMs = result.elapsedMs)) }
        val alert = frame.kind == FrameKind.ALERT || frame.priority == Priority.ALERT
        player.enqueue(PcmAudioPlayer.Request(
            samples = result.samples,
            sampleRate = result.sampleRate,
            alert = alert,
            onFirstAudio = {
                val receiveToAudio = SystemClock.elapsedRealtime() - receivedAt
                update { state -> state.copy(metrics = state.metrics.copy(endToFirstAudioMs = receiveToAudio)) }
            },
            onFinished = { update { it.copy(isSpeaking = false) } },
        ))
    }

    private suspend fun sendAck(sequence: UInt, status: Int) {
        val frame = TantraFrame(
            FrameKind.ACK, Language.UNKNOWN, Priority.NORMAL, sessionId,
            this.sequence.getAndIncrement().toUInt(), SystemClock.elapsedRealtime().toULong(),
            AckCodec.encode(sequence, status), 0,
        )
        runCatching { transport?.send(AksharaCodec.encode(frame)) }
    }

    fun useLoopback() = installTransport(LoopbackTransport(), LinkType.LOOPBACK)

    fun hostLan(port: Int = 47821) {
        val lan = TcpLanTransport(scope)
        installTransport(lan, LinkType.WIFI_LAN)
        lan.host(port)
    }

    fun connectLan(host: String, port: Int = 47821) {
        val lan = TcpLanTransport(scope)
        installTransport(lan, LinkType.WIFI_LAN)
        lan.connect(host, port)
    }

    fun hostBluetooth() {
        val adapter = context.getSystemService(BluetoothManager::class.java).adapter
        if (adapter == null) { update { it.copy(diagnostic = "Bluetooth is unavailable") }; return }
        val bluetooth = BluetoothRfcommTransport(adapter, scope)
        installTransport(bluetooth, LinkType.BLUETOOTH)
        bluetooth.host()
    }

    fun connectBluetooth(address: String) {
        val adapter = context.getSystemService(BluetoothManager::class.java).adapter
        if (adapter == null) { update { it.copy(diagnostic = "Bluetooth is unavailable") }; return }
        val device = runCatching { adapter.getRemoteDevice(address) }.getOrElse {
            update { state -> state.copy(diagnostic = "Invalid Bluetooth address") }; return
        }
        val bluetooth = BluetoothRfcommTransport(adapter, scope)
        installTransport(bluetooth, LinkType.BLUETOOTH)
        bluetooth.connect(device)
    }

    private fun installTransport(candidate: DuplexTransport, type: LinkType) {
        transport?.close(); reliableSender.clear()
        transport = candidate
        candidate.setReceiver(::receive)
        update { it.copy(linkType = type, linkState = candidate.state.value, peerLabel = candidate.peerLabel.value) }
        scope.launch { candidate.state.collect { value -> update { it.copy(linkState = value) } } }
        scope.launch { candidate.peerLabel.collect { value -> update { it.copy(peerLabel = value) } } }
    }

    private fun appendPcm(samples: ShortArray) {
        synchronized(audioBuffer) {
            samples.forEach { value ->
                audioBuffer.write(value.toInt() and 0xFF)
                audioBuffer.write((value.toInt() ushr 8) and 0xFF)
            }
        }
    }

    private fun pcmSnapshot(): ShortArray = synchronized(audioBuffer) {
        val bytes = audioBuffer.toByteArray()
        ShortArray(bytes.size / 2) { index ->
            val low = bytes[index * 2].toInt() and 0xFF
            val high = bytes[index * 2 + 1].toInt()
            ((high shl 8) or low).toShort()
        }
    }

    private inline fun update(transform: (TantraUiState) -> TantraUiState) {
        mutableState.value = transform(mutableState.value)
    }

    override fun close() {
        capture.close(); player.close(); transport?.close(); stt?.close(); tts?.close(); scope.cancel()
    }
}
