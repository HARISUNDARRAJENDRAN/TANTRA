package in.tantra.transceiver.transport

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothServerSocket
import android.bluetooth.BluetoothSocket
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
import java.util.UUID

@SuppressLint("MissingPermission")
class BluetoothRfcommTransport(
    private val adapter: BluetoothAdapter,
    private val scope: CoroutineScope,
) : DuplexTransport {
    companion object {
        val SERVICE_UUID: UUID = UUID.fromString("d59a4a15-bd5e-4e37-8be1-914dc59722a9")
        private const val SERVICE_NAME = "TANTRA"
    }

    private val mutableState = MutableStateFlow(LinkState.DISCONNECTED)
    private val mutablePeer = MutableStateFlow("No peer")
    private val sendMutex = Mutex()
    private var receiver: suspend (ByteArray) -> Unit = {}
    private var server: BluetoothServerSocket? = null
    private var socket: BluetoothSocket? = null
    private var output: DataOutputStream? = null
    private var job: Job? = null

    override val state: StateFlow<LinkState> = mutableState
    override val peerLabel: StateFlow<String> = mutablePeer

    fun bondedDevices(): Set<BluetoothDevice> = adapter.bondedDevices.orEmpty()

    fun host() {
        close()
        mutableState.value = LinkState.LISTENING
        job = scope.launch(Dispatchers.IO) {
            try {
                server = adapter.listenUsingRfcommWithServiceRecord(SERVICE_NAME, SERVICE_UUID)
                attach(server!!.accept())
            } catch (_: Throwable) {
                if (isActive) mutableState.value = LinkState.ERROR
            }
        }
    }

    fun connect(device: BluetoothDevice) {
        close()
        mutableState.value = LinkState.CONNECTING
        job = scope.launch(Dispatchers.IO) {
            try {
                adapter.cancelDiscovery()
                val candidate = device.createRfcommSocketToServiceRecord(SERVICE_UUID)
                candidate.connect()
                attach(candidate)
            } catch (_: Throwable) {
                if (isActive) mutableState.value = LinkState.ERROR
            }
        }
    }

    private suspend fun attach(connected: BluetoothSocket) {
        socket = connected
        server?.close(); server = null
        output = DataOutputStream(connected.outputStream.buffered())
        val input = DataInputStream(connected.inputStream.buffered())
        mutablePeer.value = connected.remoteDevice.name ?: connected.remoteDevice.address
        mutableState.value = LinkState.CONNECTED
        try {
            while (scope.isActive && connected.isConnected) receiver(input.readFramed())
        } finally {
            close()
        }
    }

    override suspend fun send(frame: ByteArray) = sendMutex.withLock {
        (output ?: error("Bluetooth transport is not connected")).writeFramed(frame)
    }

    override fun setReceiver(receiver: suspend (ByteArray) -> Unit) { this.receiver = receiver }

    override fun close() {
        job?.cancel(); job = null
        runCatching { output?.close() }; output = null
        runCatching { socket?.close() }; socket = null
        runCatching { server?.close() }; server = null
        mutableState.value = LinkState.DISCONNECTED
        mutablePeer.value = "No peer"
    }
}
