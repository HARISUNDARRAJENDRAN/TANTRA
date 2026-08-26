package in.tantra.transceiver.transport

import in.tantra.transceiver.model.LinkState
import kotlinx.coroutines.flow.StateFlow
import java.io.Closeable

interface DuplexTransport : Closeable {
    val state: StateFlow<LinkState>
    val peerLabel: StateFlow<String>
    suspend fun send(frame: ByteArray)
    fun setReceiver(receiver: suspend (ByteArray) -> Unit)
}
