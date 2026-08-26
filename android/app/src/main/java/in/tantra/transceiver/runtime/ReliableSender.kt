package in.tantra.transceiver.runtime

import in.tantra.transceiver.protocol.TantraFrame
import in.tantra.transceiver.transport.DuplexTransport
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap

class ReliableSender(private val scope: CoroutineScope, private val transport: () -> DuplexTransport?) {
    private val pending = ConcurrentHashMap<UInt, ByteArray>()

    suspend fun send(frame: TantraFrame, encoded: ByteArray, maxAttempts: Int) {
        pending[frame.sequence] = encoded
        transport()?.send(encoded) ?: error("No active transport")
        scope.launch {
            repeat(maxAttempts - 1) { attempt ->
                delay((350L shl attempt.coerceAtMost(2)))
                val retry = pending[frame.sequence] ?: return@launch
                runCatching { transport()?.send(retry) }
            }
            pending.remove(frame.sequence)
        }
    }

    fun acknowledge(sequence: UInt) { pending.remove(sequence) }
    fun clear() = pending.clear()
}
