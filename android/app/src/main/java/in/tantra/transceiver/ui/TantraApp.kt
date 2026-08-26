package in.tantra.transceiver.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import in.tantra.transceiver.model.Language
import in.tantra.transceiver.model.LinkState
import in.tantra.transceiver.model.SessionMode
import in.tantra.transceiver.runtime.TantraRuntime
import kotlinx.coroutines.launch

private val TantraColors = lightColorScheme(
    primary = androidx.compose.ui.graphics.Color(0xFF405FE6),
    onPrimary = androidx.compose.ui.graphics.Color.White,
    background = androidx.compose.ui.graphics.Color(0xFFF7F6F2),
    surface = androidx.compose.ui.graphics.Color(0xFFFFFFFF),
    onBackground = androidx.compose.ui.graphics.Color(0xFF101113),
    onSurface = androidx.compose.ui.graphics.Color(0xFF101113),
    outline = androidx.compose.ui.graphics.Color(0xFFD9D7D0),
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TantraApp(runtime: TantraRuntime) {
    val state by runtime.state.collectAsState()
    val scope = rememberCoroutineScope()
    var host by remember { mutableStateOf("192.168.1.2") }
    var bluetoothAddress by remember { mutableStateOf("") }
    val packPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch { runtime.importModelPack(uri) }
    }

    MaterialTheme(colorScheme = TantraColors) {
        Surface(Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column {
                        Text("TANTRA", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Black)
                        Text("Offline token-native speech link", style = MaterialTheme.typography.bodyMedium)
                    }
                    StatusPill(state.linkState, state.peerLabel)
                }

                Card(shape = RoundedCornerShape(24.dp)) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Model", fontWeight = FontWeight.SemiBold)
                        Text(state.modelPackId ?: "No verified model pack loaded")
                        Button(onClick = { packPicker.launch(arrayOf("application/zip", "application/octet-stream")) }) {
                            Text("Import .tantra-pack")
                        }
                        state.diagnostic?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                    }
                }

                Card(shape = RoundedCornerShape(24.dp)) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Language", fontWeight = FontWeight.SemiBold)
                        LanguageSelector(state.language, runtime::setLanguage)
                        Text("Session", fontWeight = FontWeight.SemiBold)
                        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                            SessionMode.entries.forEachIndexed { index, mode ->
                                SegmentedButton(
                                    selected = state.mode == mode,
                                    onClick = { runtime.setMode(mode) },
                                    shape = SegmentedButtonDefaults.itemShape(index, SessionMode.entries.size),
                                    label = { Text(if (mode == SessionMode.PUSH_TO_TALK) "Push to talk" else "Continuous") },
                                )
                            }
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Switch(checked = state.alertMode, onCheckedChange = runtime::setAlertMode)
                            Spacer(Modifier.width(10.dp))
                            Column {
                                Text("Alert priority", fontWeight = FontWeight.Medium)
                                Text("Requests alarm focus and retry-until-ack delivery", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }

                Card(shape = RoundedCornerShape(24.dp)) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Peer link", fontWeight = FontWeight.SemiBold)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = runtime::useLoopback) { Text("Loopback") }
                            OutlinedButton(onClick = { runtime.hostLan() }) { Text("Host Wi-Fi") }
                            OutlinedButton(onClick = runtime::hostBluetooth) { Text("Host Bluetooth") }
                        }
                        OutlinedTextField(host, { host = it }, label = { Text("Peer IPv4 / hostname") }, modifier = Modifier.fillMaxWidth())
                        Button(onClick = { runtime.connectLan(host) }, modifier = Modifier.fillMaxWidth()) { Text("Connect over Wi-Fi LAN") }
                        OutlinedTextField(bluetoothAddress, { bluetoothAddress = it }, label = { Text("Bonded Bluetooth MAC") }, modifier = Modifier.fillMaxWidth())
                        OutlinedButton(onClick = { runtime.connectBluetooth(bluetoothAddress) }, modifier = Modifier.fillMaxWidth()) { Text("Connect bonded device") }
                    }
                }

                if (state.mode == SessionMode.PUSH_TO_TALK) {
                    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                        Surface(
                            modifier = Modifier.size(180.dp).pointerInput(state.modelReady) {
                                detectTapGestures(
                                    onPress = {
                                        runtime.startPtt()
                                        tryAwaitRelease()
                                        runtime.stopPtt()
                                    }
                                )
                            },
                            shape = CircleShape,
                            color = if (state.isCapturing) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                            tonalElevation = 8.dp,
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text(if (state.isCapturing) "LISTENING" else "HOLD TO TALK", color = androidx.compose.ui.graphics.Color.White, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }

                TranscriptCard("You", state.localTranscript.ifBlank { "Your recognized speech appears here." })
                TranscriptCard("Peer", state.remoteTranscript.ifBlank { "Received speech appears here before local playback." })

                Card(shape = RoundedCornerShape(24.dp)) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Last measured path", fontWeight = FontWeight.SemiBold)
                        MetricRow("ASR inference", state.metrics.asrLastMs)
                        MetricRow("TTS inference", state.metrics.ttsLastMs)
                        MetricRow("Receive → first audio", state.metrics.endToFirstAudioMs)
                        Text("${state.metrics.bytesSent} B sent · ${state.metrics.bytesReceived} B received · ${state.metrics.droppedFrames} dropped", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusPill(state: LinkState, peer: String) {
    val text = when (state) {
        LinkState.CONNECTED -> "Connected · $peer"
        LinkState.LISTENING -> "Listening"
        LinkState.CONNECTING -> "Connecting"
        LinkState.ERROR -> "Link error"
        LinkState.DISCONNECTED -> "Offline"
    }
    Surface(shape = RoundedCornerShape(999.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Text(text, Modifier.padding(horizontal = 12.dp, vertical = 8.dp), style = MaterialTheme.typography.labelMedium)
    }
}

@Composable
private fun LanguageSelector(current: Language, onSelect: (Language) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Box {
        OutlinedButton(onClick = { expanded = true }, modifier = Modifier.fillMaxWidth()) { Text(current.displayName) }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            Language.entries.filter { it != Language.UNKNOWN }.forEach { language ->
                DropdownMenuItem(text = { Text(language.displayName) }, onClick = { onSelect(language); expanded = false })
            }
        }
    }
}

@Composable
private fun TranscriptCard(label: String, text: String) {
    Card(shape = RoundedCornerShape(24.dp)) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Text(text, style = MaterialTheme.typography.bodyLarge)
        }
    }
}

@Composable
private fun MetricRow(label: String, value: Long?) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label)
        Text(value?.let { "$it ms" } ?: "—", fontWeight = FontWeight.Medium)
    }
}
