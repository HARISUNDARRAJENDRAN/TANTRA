package in.tantra.transceiver.ml

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

class ModelPackManager(private val context: Context) {
    private val json = Json { ignoreUnknownKeys = false }
    private val root = File(context.filesDir, "modelpacks").apply { mkdirs() }

    suspend fun import(uri: Uri): LoadedModelPack = withContext(Dispatchers.IO) {
        val staging = File(root, ".staging-${System.nanoTime()}").apply { mkdirs() }
        try {
            val rootPath = staging.canonicalPath + File.separator
            context.contentResolver.openInputStream(uri)?.use { source ->
                ZipInputStream(source.buffered()).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (entry.isDirectory) continue
                        val target = File(staging, entry.name)
                        if (!target.canonicalPath.startsWith(rootPath)) error("Unsafe model-pack path")
                        target.parentFile?.mkdirs()
                        target.outputStream().buffered().use { zip.copyTo(it) }
                    }
                }
            } ?: error("Unable to open model pack")
            val loaded = loadDirectory(staging)
            val destination = File(root, loaded.manifest.packId)
            if (destination.exists()) destination.deleteRecursively()
            check(staging.renameTo(destination)) { "Could not activate model pack" }
            loadDirectory(destination)
        } catch (failure: Throwable) {
            staging.deleteRecursively()
            throw failure
        }
    }

    fun loadActive(packId: String): LoadedModelPack = loadDirectory(File(root, packId))

    fun availablePackIds(): List<String> = root.listFiles()
        ?.filter { it.isDirectory && !it.name.startsWith('.') }
        ?.map { it.name }
        ?.sorted()
        .orEmpty()

    private fun loadDirectory(directory: File): LoadedModelPack {
        val manifestFile = File(directory, "manifest.json")
        val modelCard = File(directory, "MODEL_CARD.md")
        val licenseDir = File(directory, "LICENSES")
        require(manifestFile.isFile) { "Model pack is missing manifest.json" }
        require(modelCard.isFile) { "Model pack is missing MODEL_CARD.md" }
        require(licenseDir.isDirectory && !licenseDir.listFiles().isNullOrEmpty()) { "Model pack has no license inventory" }
        val manifest = json.decodeFromString<ModelPackManifest>(manifestFile.readText())
        require(manifest.formatVersion == 1) { "Unsupported model-pack version" }
        require(manifest.packId.matches(Regex("[a-zA-Z0-9._-]{1,80}"))) { "Unsafe pack id" }
        require(manifest.licenseSpdx.isNotEmpty()) { "Missing model license metadata" }
        manifest.files.forEach { (relative, expected) ->
            val file = File(directory, relative)
            require(file.isFile) { "Missing model file: $relative" }
            require(sha256(file).equals(expected, ignoreCase = true)) { "Hash mismatch: $relative" }
        }
        val vocabularyFile = File(directory, "vocab.json")
        require(vocabularyFile.isFile) { "Missing vocab.json" }
        require(sha256(vocabularyFile).equals(manifest.vocabSha256, ignoreCase = true)) { "Vocabulary hash mismatch" }
        val vocabulary = Vocabulary(json.decodeFromString<VocabularyFile>(vocabularyFile.readText()))
        return LoadedModelPack(directory, manifest, vocabulary)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
