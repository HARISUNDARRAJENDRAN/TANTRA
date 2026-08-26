package in.tantra.transceiver.protocol

import in.tantra.transceiver.model.Language
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

object AksharaCodec {
    private const val VERSION = 1
    private const val HEADER_SIZE = 23
    private const val CRC_SIZE = 4
    private val magic = byteArrayOf('T'.code.toByte(), 'N'.code.toByte())

    fun encode(frame: TantraFrame): ByteArray {
        require(frame.flags and 0x1F == frame.flags) { "Flags exceed five bits" }
        require(frame.payload.size <= 65_535) { "Payload exceeds uint16" }
        val buffer = ByteBuffer.allocate(HEADER_SIZE + frame.payload.size + CRC_SIZE)
            .order(ByteOrder.BIG_ENDIAN)
        buffer.put(magic)
        buffer.put(((VERSION shl 5) or frame.flags).toByte())
        buffer.put(((frame.kind.wire shl 4) or frame.language.wireId).toByte())
        buffer.put(frame.priority.wire.toByte())
        buffer.putInt(frame.sessionId.toInt())
        buffer.putInt(frame.sequence.toInt())
        buffer.putLong(frame.senderTimestampMs.toLong())
        buffer.putShort(frame.payload.size.toShort())
        buffer.put(frame.payload)
        val crc = CRC32().apply { update(buffer.array(), 0, HEADER_SIZE + frame.payload.size) }.value
        buffer.putInt(crc.toInt())
        return buffer.array()
    }

    fun decode(bytes: ByteArray): TantraFrame {
        if (bytes.size < HEADER_SIZE + CRC_SIZE) throw ProtocolException("Frame too short")
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
        if (buffer.get() != magic[0] || buffer.get() != magic[1]) throw ProtocolException("Bad magic")
        val versionFlags = buffer.get().toInt() and 0xFF
        if (versionFlags ushr 5 != VERSION) throw ProtocolException("Unsupported version")
        val kindLanguage = buffer.get().toInt() and 0xFF
        val priorityWire = buffer.get().toInt() and 0xFF
        val sessionId = buffer.int.toUInt()
        val sequence = buffer.int.toUInt()
        val timestamp = buffer.long.toULong()
        val payloadSize = buffer.short.toInt() and 0xFFFF
        val expected = HEADER_SIZE + payloadSize + CRC_SIZE
        if (bytes.size != expected) throw ProtocolException("Length mismatch")
        val expectedCrc = ByteBuffer.wrap(bytes, bytes.size - CRC_SIZE, CRC_SIZE)
            .order(ByteOrder.BIG_ENDIAN).int.toUInt().toLong()
        val actualCrc = CRC32().apply { update(bytes, 0, bytes.size - CRC_SIZE) }.value
        if (expectedCrc != actualCrc) throw ProtocolException("CRC mismatch")
        val payload = ByteArray(payloadSize).also { buffer.get(it) }
        val kind = FrameKind.fromWire(kindLanguage ushr 4) ?: throw ProtocolException("Unknown kind")
        val language = Language.fromWireId(kindLanguage and 0x0F)
        val priority = Priority.entries.firstOrNull { it.wire == priorityWire }
            ?: throw ProtocolException("Unknown priority")
        return TantraFrame(kind, language, priority, sessionId, sequence, timestamp, payload, versionFlags and 0x1F)
    }
}

data class TextDelta(val baseSequence: UInt, val replaceFrom: Int, val suffixTokens: List<Int>)

object TextDeltaCodec {
    fun encode(delta: TextDelta): ByteArray {
        require(delta.replaceFrom >= 0)
        val out = ByteArrayOutputStream()
        out.write(ByteBuffer.allocate(4).order(ByteOrder.BIG_ENDIAN).putInt(delta.baseSequence.toInt()).array())
        writeVarUInt(out, delta.replaceFrom.toLong())
        writeVarUInt(out, delta.suffixTokens.size.toLong())
        delta.suffixTokens.forEach { require(it >= 0); writeVarUInt(out, it.toLong()) }
        return out.toByteArray()
    }

    fun decode(bytes: ByteArray): TextDelta {
        if (bytes.size < 4) throw ProtocolException("Token delta missing base sequence")
        val base = ByteBuffer.wrap(bytes, 0, 4).order(ByteOrder.BIG_ENDIAN).int.toUInt()
        var offset = 4
        val replace = readVarUInt(bytes, offset).also { offset = it.second }.first.toInt()
        val count = readVarUInt(bytes, offset).also { offset = it.second }.first.toInt()
        val tokens = ArrayList<Int>(count)
        repeat(count) {
            val (token, next) = readVarUInt(bytes, offset)
            offset = next
            tokens += token.toInt()
        }
        if (offset != bytes.size) throw ProtocolException("Trailing token bytes")
        return TextDelta(base, replace, tokens)
    }

    fun make(previous: List<Int>, current: List<Int>, baseSequence: UInt): TextDelta {
        var common = 0
        while (common < previous.size && common < current.size && previous[common] == current[common]) common++
        return TextDelta(baseSequence, common, current.drop(common))
    }

    fun apply(previous: List<Int>, delta: TextDelta): List<Int> {
        if (delta.replaceFrom > previous.size) throw ProtocolException("Bad delta base")
        return previous.take(delta.replaceFrom) + delta.suffixTokens
    }

    private fun writeVarUInt(out: ByteArrayOutputStream, initial: Long) {
        var value = initial
        do {
            var byte = (value and 0x7F).toInt()
            value = value ushr 7
            if (value != 0L) byte = byte or 0x80
            out.write(byte)
        } while (value != 0L)
    }

    private fun readVarUInt(bytes: ByteArray, start: Int): Pair<Long, Int> {
        var value = 0L
        var shift = 0
        var offset = start
        while (offset < bytes.size && shift < 64) {
            val byte = bytes[offset++].toInt() and 0xFF
            value = value or ((byte and 0x7F).toLong() shl shift)
            if (byte and 0x80 == 0) return value to offset
            shift += 7
        }
        throw ProtocolException("Truncated varuint")
    }
}

object AckCodec {
    fun encode(sequence: UInt, status: Int = 0): ByteArray = ByteBuffer.allocate(5)
        .order(ByteOrder.BIG_ENDIAN).putInt(sequence.toInt()).put(status.toByte()).array()

    fun decode(payload: ByteArray): Pair<UInt, Int> {
        if (payload.size != 5) throw ProtocolException("ACK must contain five bytes")
        val buffer = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
        return buffer.int.toUInt() to (buffer.get().toInt() and 0xFF)
    }
}
