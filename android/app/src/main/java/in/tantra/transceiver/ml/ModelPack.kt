package in.tantra.transceiver.ml

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class AsrContract(
    @SerialName("samples_input") val samplesInput: String = "samples",
    @SerialName("lengths_input") val lengthsInput: String = "sample_lengths",
    @SerialName("language_input") val languageInput: String? = "language_id",
    @SerialName("logits_output") val logitsOutput: String = "logits",
    @SerialName("blank_id") val blankId: Int = 0,
)

@Serializable
data class TtsContract(
    @SerialName("tokens_input") val tokensInput: String = "tokens",
    @SerialName("lengths_input") val lengthsInput: String = "token_lengths",
    @SerialName("language_input") val languageInput: String = "language_id",
    @SerialName("speaker_input") val speakerInput: String = "speaker_id",
    @SerialName("speed_input") val speedInput: String = "speed",
    @SerialName("audio_output") val audioOutput: String = "audio",
)

@Serializable
data class ModelPackManifest(
    @SerialName("format_version") val formatVersion: Int,
    @SerialName("pack_id") val packId: String,
    val languages: List<String>,
    @SerialName("sample_rate") val sampleRate: Int = 16_000,
    @SerialName("tts_sample_rate") val ttsSampleRate: Int = 22_050,
    @SerialName("vocab_sha256") val vocabSha256: String,
    val files: Map<String, String>,
    @SerialName("license_spdx") val licenseSpdx: List<String>,
    val asr: AsrContract = AsrContract(),
    val tts: TtsContract? = null,
)

@Serializable
data class VocabularyFile(
    val tokens: List<String>,
    @SerialName("unknown_id") val unknownId: Int = 1,
    @SerialName("bos_id") val bosId: Int? = null,
    @SerialName("eos_id") val eosId: Int? = null,
)

data class LoadedModelPack(
    val directory: java.io.File,
    val manifest: ModelPackManifest,
    val vocabulary: Vocabulary,
)
