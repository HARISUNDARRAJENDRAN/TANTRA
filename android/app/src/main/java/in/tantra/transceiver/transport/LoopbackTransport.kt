package in.tantra.transceiver.transport

import in.tantra.transceiver.model.LinkState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class LoopbackTransport(private val latencyMs: Long = 15) : DuplexTransport {
    private val mutableState = MutableStateFlow(LinkState.CONNECTED)
    private val mutablePeer = MutableStateFlow("Local loopback")
    private var receiver: suspend (ByteArray) -> Unit = {}
    override val state: StateFlow<LinkState> = mutableState
    override val peerLabel: StateFlow<String> = mutablePeer

    override suspend fun send(frame: ByteArray) {
        delay(latencyMs)
        receiver(frame.copyOf())
    }

    override fun setReceiver(receiver: suspend (ByteArray) -> Unit) { this.receiver = receiver }
    override fun close() { mutableState.value = LinkState.DISCONNECTED }
}
