package in.tantra.transceiver.model

enum class SessionMode { PUSH_TO_TALK, CONTINUOUS }
enum class LinkType { LOOPBACK, WIFI_LAN, BLUETOOTH }
enum class LinkState { DISCONNECTED, LISTENING, CONNECTING, CONNECTED, ERROR }

data class RuntimeMetrics(
    val asrLastMs: Long? = null,
    val ttsLastMs: Long? = null,
    val networkLastMs: Long? = null,
    val endToFirstAudioMs: Long? = null,
    val bytesSent: Long = 0,
    val bytesReceived: Long = 0,
    val droppedFrames: Long = 0,
)

data class TantraUiState(
    val language: Language = Language.HINDI,
    val mode: SessionMode = SessionMode.PUSH_TO_TALK,
    val linkType: LinkType = LinkType.LOOPBACK,
    val linkState: LinkState = LinkState.DISCONNECTED,
    val peerLabel: String = "No peer",
    val isCapturing: Boolean = false,
    val isSpeaking: Boolean = false,
    val alertMode: Boolean = false,
    val modelPackId: String? = null,
    val modelReady: Boolean = false,
    val localTranscript: String = "",
    val remoteTranscript: String = "",
    val diagnostic: String? = null,
    val metrics: RuntimeMetrics = RuntimeMetrics(),
)
