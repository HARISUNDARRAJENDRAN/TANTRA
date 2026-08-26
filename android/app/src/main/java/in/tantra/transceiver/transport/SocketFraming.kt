package in.tantra.transceiver.transport

import java.io.DataInputStream
import java.io.DataOutputStream

internal const val MAX_FRAME_BYTES = 70_000

internal fun DataInputStream.readFramed(): ByteArray {
    val size = readInt()
    require(size in 1..MAX_FRAME_BYTES) { "Invalid framed length $size" }
    return ByteArray(size).also(::readFully)
}

internal fun DataOutputStream.writeFramed(bytes: ByteArray) {
    require(bytes.size in 1..MAX_FRAME_BYTES)
    writeInt(bytes.size)
    write(bytes)
    flush()
}
