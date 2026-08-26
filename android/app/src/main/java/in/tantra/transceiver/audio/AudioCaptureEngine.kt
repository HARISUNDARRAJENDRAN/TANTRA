package in.tantra.transceiver.audio

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AudioCaptureEngine(private val context: Context, private val sampleRate: Int = 16_000) : AutoCloseable {
    private var record: AudioRecord? = null
    private var job: Job? = null
    private val effects = mutableListOf<android.media.audiofx.AudioEffect>()

    @SuppressLint("MissingPermission")
    fun start(scope: CoroutineScope, onFrame: suspend (ShortArray) -> Unit) {
        if (job != null) return
        check(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            "Microphone permission not granted"
        }
        val frameSamples = sampleRate / 50
        val minimum = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        check(minimum > 0) { "AudioRecord configuration unsupported: $minimum" }
        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minimum * 2, frameSamples * 8),
        )
        check(audioRecord.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialize" }
        record = audioRecord
        enableEffects(audioRecord.audioSessionId)
        audioRecord.startRecording()
        job = scope.launch(Dispatchers.IO) {
            val frame = ShortArray(frameSamples)
            while (isActive) {
                val count = audioRecord.read(frame, 0, frame.size, AudioRecord.READ_BLOCKING)
                if (count > 0) onFrame(if (count == frame.size) frame.copyOf() else frame.copyOf(count))
            }
        }
    }

    suspend fun stop() {
        val activeJob = job ?: return
        job = null
        activeJob.cancel()
        withContext(Dispatchers.IO) {
            runCatching { record?.stop() }
            record?.release()
            record = null
            effects.forEach { runCatching { it.release() } }
            effects.clear()
        }
    }

    override fun close() {
        job?.cancel()
        runCatching { record?.stop() }
        record?.release()
        effects.forEach { runCatching { it.release() } }
        effects.clear()
        record = null
        job = null
    }

    private fun enableEffects(sessionId: Int) {
        fun add(effect: android.media.audiofx.AudioEffect?) {
            if (effect != null) {
                runCatching { effect.enabled = true }
                effects += effect
            }
        }
        if (AcousticEchoCanceler.isAvailable()) add(AcousticEchoCanceler.create(sessionId))
        if (NoiseSuppressor.isAvailable()) add(NoiseSuppressor.create(sessionId))
        if (AutomaticGainControl.isAvailable()) add(AutomaticGainControl.create(sessionId))
    }
}
