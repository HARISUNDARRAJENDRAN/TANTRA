package in.tantra.transceiver.protocol

import in.tantra.transceiver.model.Language
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AksharaCodecTest {
    @Test fun frameRoundTrip() {
        val frame = TantraFrame(
            FrameKind.ALERT, Language.HINDI, Priority.ALERT, 42u, 7u, 1234u,
            "सहायता".encodeToByteArray(), FrameFlags.FINAL or FrameFlags.ACK_REQUIRED,
        )
        assertEquals(frame, AksharaCodec.decode(AksharaCodec.encode(frame)))
    }

    @Test fun deltaRoundTrip() {
        val previous = listOf(1, 2, 3, 4)
        val current = listOf(1, 2, 8, 9)
        val delta = TextDeltaCodec.make(previous, current, 5u)
        val decoded = TextDeltaCodec.decode(TextDeltaCodec.encode(delta))
        assertEquals(current, TextDeltaCodec.apply(previous, decoded))
    }

    @Test fun compactIndicPayload() {
        val ids = List(12) { it + 1 }
        val encoded = TextDeltaCodec.encode(TextDelta(0u, 0, ids))
        assertTrue(encoded.size < "आप सुरक्षित हैं".encodeToByteArray().size + 8)
    }
}
