package in.tantra.transceiver.audio

import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveEndpointDetectorTest {
    @Test fun startsAndEndsSpeech() {
        val detector = AdaptiveEndpointDetector()
        val silence = ShortArray(detector.expectedFrameSamples)
        val speech = ShortArray(detector.expectedFrameSamples) { if (it % 2 == 0) 9000 else -9000 }
        repeat(20) { detector.accept(silence) }
        val events = List(4) { detector.accept(speech) }
        assertTrue(events.contains(AdaptiveEndpointDetector.Event.SPEECH_START))
        val endingEvents = List(40) { detector.accept(silence) }
        assertTrue(endingEvents.contains(AdaptiveEndpointDetector.Event.SPEECH_END))
    }
}
