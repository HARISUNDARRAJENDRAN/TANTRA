package in.tantra.transceiver.transport

import in.tantra.transceiver.model.LinkState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket

class TcpLanTransport(private val scope: CoroutineScope) : DuplexTransport {
    private val mutableState = MutableStateFlow(LinkState.DISCONNECTED)
    private val mutablePeer = MutableStateFlow("No peer")
    private val sendMutex = Mutex()
    private var receiver: suspend (ByteArray) -> Unit = {}
    private var server: ServerSocket? = null
    private var socket: Socket? = null
    private var inputJob: Job? = null
    private var output: DataOutputStream? = null

    override val state: StateFlow<LinkState> = mutableState
    override val peerLabel: StateFlow<String> = mutablePeer

    fun host(port: Int = 47821) {
        close()
        mutableState.value = LinkState.LISTENING
        inputJob = scope.launch(Dispatchers.IO) {
            try {
                server = ServerSocket(port).apply { reuseAddress = true }
                attach(server!!.accept())
            } catch (failure: Throwable) {
                if (isActive) mutableState.value = LinkState.ERROR
            }
        }
    }

    fun connect(host: String, port: Int = 47821) {
        close()
        mutableState.value = LinkState.CONNECTING
        inputJob = scope.launch(Dispatchers.IO) {
            try {
                attach(Socket().apply {
                    tcpNoDelay = true
                    keepAlive = true
                    connect(InetSocketAddress(host, port), 8_000)
                })
            } catch (failure: Throwable) {
                if (isActive) mutableState.value = LinkState.ERROR
            }
        }
    }

    private suspend fun attach(connected: Socket) {
        socket = connected.apply { tcpNoDelay = true; keepAlive = true }
        server?.close(); server = null
        output = DataOutputStream(connected.getOutputStream().buffered())
        val input = DataInputStream(connected.getInputStream().buffered())
        mutablePeer.value = connected.inetAddress.hostAddress ?: connected.inetAddress.hostName
        mutableState.value = LinkState.CONNECTED
        try {
            while (scope.isActive && !connected.isClosed) receiver(input.readFramed())
        } finally {
            close()
        }
    }

    override suspend fun send(frame: ByteArray) = sendMutex.withLock {
        val active = output ?: error("LAN transport is not connected")
        active.writeFramed(frame)
    }

    override fun setReceiver(receiver: suspend (ByteArray) -> Unit) { this.receiver = receiver }

    override fun close() {
        inputJob?.cancel(); inputJob = null
        runCatching { output?.close() }; output = null
        runCatching { socket?.close() }; socket = null
        runCatching { server?.close() }; server = null
        mutableState.value = LinkState.DISCONNECTED
        mutablePeer.value = "No peer"
    }
}
