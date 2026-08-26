package in.tantra.transceiver.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import in.tantra.transceiver.model.Language
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.system.measureTimeMillis

interface SttEngine : AutoCloseable {
    data class Result(val text: String, val elapsedMs: Long)
    fun transcribe(pcm16: ShortArray, language: Language): Result
}

interface TtsEngine : AutoCloseable {
    data class Result(val samples: FloatArray, val sampleRate: Int, val elapsedMs: Long)
    fun synthesize(text: String, language: Language, speakerId: Long = 0, speed: Float = 1f): Result
}

class OnnxCtcSttEngine(private val pack: LoadedModelPack) : SttEngine {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
        }
        session = env.createSession(java.io.File(pack.directory, "asr.onnx").absolutePath, options)
    }

    override fun transcribe(pcm16: ShortArray, language: Language): SttEngine.Result {
        val contract = pack.manifest.asr
        val samples = FloatArray(pcm16.size) { pcm16[it] / 32768f }
        var text = ""
        val elapsed = measureTimeMillis {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(samples), longArrayOf(1, samples.size.toLong())).use { waveform ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(samples.size.toLong())), longArrayOf(1)).use { lengths ->
                    val inputs = linkedMapOf<String, OnnxTensor>(
                        contract.samplesInput to waveform,
                        contract.lengthsInput to lengths,
                    )
                    val languageTensor = contract.languageInput?.let {
                        OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(language.wireId.toLong())), longArrayOf(1))
                    }
                    if (languageTensor != null && contract.languageInput != null) inputs[contract.languageInput] = languageTensor
                    try {
                        session.run(inputs).use { outputs ->
                            val value = outputs.get(contract.logitsOutput).orElseThrow().value
                            @Suppress("UNCHECKED_CAST")
                            val logits = value as Array<Array<FloatArray>>
                            val tokenIds = ArrayList<Int>()
                            var previous = -1
                            for (frame in logits[0]) {
                                var best = 0
                                var bestScore = Float.NEGATIVE_INFINITY
                                for (index in frame.indices) if (frame[index] > bestScore) {
                                    best = index; bestScore = frame[index]
                                }
                                if (best != contract.blankId && best != previous) tokenIds += best
                                previous = best
                            }
                            text = pack.vocabulary.decode(tokenIds)
                        }
                    } finally {
                        languageTensor?.close()
                    }
                }
            }
        }
        return SttEngine.Result(text, elapsed)
    }

    override fun close() = session.close()
}

class OnnxTantraTtsEngine(private val pack: LoadedModelPack) : TtsEngine {
    private val env = OrtEnvironment.getEnvironment()
    private val contract = requireNotNull(pack.manifest.tts) { "Pack does not contain TTS contract" }
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
        }
        session = env.createSession(java.io.File(pack.directory, "tts.onnx").absolutePath, options)
    }

    override fun synthesize(text: String, language: Language, speakerId: Long, speed: Float): TtsEngine.Result {
        val ids = pack.vocabulary.encode(text).map(Int::toLong).toLongArray()
        var audio = FloatArray(0)
        val elapsed = measureTimeMillis {
            OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { tokens ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(ids.size.toLong())), longArrayOf(1)).use { lengths ->
                    OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(language.wireId.toLong())), longArrayOf(1)).use { lang ->
                        OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(speakerId)), longArrayOf(1)).use { speaker ->
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1)).use { speedTensor ->
                                session.run(mapOf(
                                    contract.tokensInput to tokens,
                                    contract.lengthsInput to lengths,
                                    contract.languageInput to lang,
                                    contract.speakerInput to speaker,
                                    contract.speedInput to speedTensor,
                                )).use { outputs ->
                                    val value = outputs.get(contract.audioOutput).orElseThrow().value
                                    audio = when (value) {
                                        is FloatArray -> value
                                        is Array<*> -> when (val first = value.firstOrNull()) {
                                            is FloatArray -> first
                                            is Array<*> -> (first.firstOrNull() as? FloatArray) ?: error("Unsupported TTS tensor shape")
                                            else -> error("Unsupported TTS tensor type")
                                        }
                                        else -> error("Unsupported TTS output")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return TtsEngine.Result(audio, pack.manifest.ttsSampleRate, elapsed)
    }

    override fun close() = session.close()
}
