package in.tantra.transceiver.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PcmAudioPlayer(context: Context, scope: CoroutineScope) : AutoCloseable {
    data class Request(
        val samples: FloatArray,
        val sampleRate: Int,
        val alert: Boolean,
        val onFirstAudio: (() -> Unit)? = null,
        val onFinished: (() -> Unit)? = null,
    )

    private val audioManager = context.getSystemService(AudioManager::class.java)
    private val queue = Channel<Request>(Channel.UNLIMITED)
    private val worker = scope.launch(Dispatchers.IO) {
        for (request in queue) playNow(request)
    }

    suspend fun enqueue(request: Request) = queue.send(request)

    private suspend fun playNow(request: Request) = withContext(Dispatchers.IO) {
        val usage = if (request.alert) AudioAttributes.USAGE_ALARM else AudioAttributes.USAGE_VOICE_COMMUNICATION
        val attributes = AudioAttributes.Builder()
            .setUsage(usage)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        val format = AudioFormat.Builder()
            .setSampleRate(request.sampleRate)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        val pcm = ShortArray(request.samples.size) { index ->
            (request.samples[index].coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort()
        }
        val minimum = AudioTrack.getMinBufferSize(request.sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(attributes)
            .setAudioFormat(format)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(minimum, pcm.size.coerceAtMost(request.sampleRate) * 2))
            .build()
        val focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
            .setAudioAttributes(attributes)
            .setAcceptsDelayedFocusGain(false)
            .build()
        val alarmStream = AudioManager.STREAM_ALARM
        val previousVolume = if (request.alert) audioManager.getStreamVolume(alarmStream) else -1
        try {
            audioManager.requestAudioFocus(focus)
            if (request.alert) {
                runCatching { audioManager.setStreamVolume(alarmStream, audioManager.getStreamMaxVolume(alarmStream), 0) }
            }
            track.play()
            request.onFirstAudio?.invoke()
            var offset = 0
            while (offset < pcm.size) {
                val wrote = track.write(pcm, offset, pcm.size - offset, AudioTrack.WRITE_BLOCKING)
                if (wrote <= 0) break
                offset += wrote
            }
            track.stop()
        } finally {
            track.release()
            audioManager.abandonAudioFocusRequest(focus)
            if (request.alert && previousVolume >= 0) {
                runCatching { audioManager.setStreamVolume(alarmStream, previousVolume, 0) }
            }
            request.onFinished?.invoke()
        }
    }

    override fun close() {
        queue.close()
        worker.cancel()
    }
}
