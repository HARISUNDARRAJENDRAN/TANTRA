package in.tantra.transceiver.protocol

import in.tantra.transceiver.model.Language

enum class FrameKind(val wire: Int) {
    HELLO(1), PARTIAL(2), CLAUSE(3), FINAL(4), ALERT(5), ACK(6), HEARTBEAT(7), CONTROL(8);
    companion object { fun fromWire(value: Int) = entries.firstOrNull { it.wire == value } }
}

enum class Priority(val wire: Int) { NORMAL(0), URGENT(1), ALERT(2) }

object FrameFlags {
    const val FINAL = 1
    const val TOKEN_PAYLOAD = 1 shl 1
    const val ACK_REQUIRED = 1 shl 2
    const val ENCRYPTED = 1 shl 3
    const val FULL_SNAPSHOT = 1 shl 4
}

data class TantraFrame(
    val kind: FrameKind,
    val language: Language,
    val priority: Priority,
    val sessionId: UInt,
    val sequence: UInt,
    val senderTimestampMs: ULong,
    val payload: ByteArray = byteArrayOf(),
    val flags: Int = 0,
) {
    override fun equals(other: Any?): Boolean = other is TantraFrame &&
        kind == other.kind && language == other.language && priority == other.priority &&
        sessionId == other.sessionId && sequence == other.sequence &&
        senderTimestampMs == other.senderTimestampMs && flags == other.flags &&
        payload.contentEquals(other.payload)

    override fun hashCode(): Int = 31 * sequence.hashCode() + payload.contentHashCode()
}

class ProtocolException(message: String) : IllegalArgumentException(message)
