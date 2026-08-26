package in.tantra.transceiver.audio

import kotlin.math.log10
import kotlin.math.sqrt

class AdaptiveEndpointDetector(
    private val sampleRate: Int = 16_000,
    private val frameMs: Int = 20,
    private val startFrames: Int = 3,
    private val endpointSilenceMs: Int = 460,
    private val hangoverMs: Int = 180,
    private val minimumSpeechMs: Int = 180,
    private val maximumUtteranceMs: Int = 12_000,
) {
    enum class Event { IDLE, SPEECH_START, SPEECH, SPEECH_END }

    private var noiseDb = -58.0
    private var speechFrames = 0
    private var silenceFrames = 0
    private var utteranceFrames = 0
    private var active = false

    val expectedFrameSamples: Int = sampleRate * frameMs / 1000

    fun accept(samples: ShortArray): Event {
        if (samples.isEmpty()) return Event.IDLE
        val rms = sqrt(samples.fold(0.0) { acc, sample ->
            val normalized = sample / 32768.0
            acc + normalized * normalized
        } / samples.size).coerceAtLeast(1e-8)
        val db = 20.0 * log10(rms)
        if (!active) noiseDb = 0.985 * noiseDb + 0.015 * db.coerceAtMost(noiseDb + 4.0)
        val threshold = (noiseDb + 11.0).coerceIn(-48.0, -25.0)
        val speech = db >= threshold

        if (!active) {
            speechFrames = if (speech) speechFrames + 1 else 0
            if (speechFrames >= startFrames) {
                active = true
                silenceFrames = 0
                utteranceFrames = speechFrames
                return Event.SPEECH_START
            }
            return Event.IDLE
        }

        utteranceFrames++
        silenceFrames = if (speech) 0 else silenceFrames + 1
        val speechMs = utteranceFrames * frameMs
        val silenceMs = silenceFrames * frameMs
        val shouldEnd = speechMs >= maximumUtteranceMs ||
            (speechMs >= minimumSpeechMs && silenceMs >= endpointSilenceMs + hangoverMs)
        return if (shouldEnd) {
            resetUtterance()
            Event.SPEECH_END
        } else Event.SPEECH
    }

    fun forceEnd(): Event = if (active) { resetUtterance(); Event.SPEECH_END } else Event.IDLE

    fun reset() {
        noiseDb = -58.0
        resetUtterance()
    }

    private fun resetUtterance() {
        speechFrames = 0
        silenceFrames = 0
        utteranceFrames = 0
        active = false
    }
}
