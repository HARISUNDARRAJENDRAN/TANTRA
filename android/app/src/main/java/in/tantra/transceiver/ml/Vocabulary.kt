package in.tantra.transceiver.ml

import java.text.Normalizer

class Vocabulary(private val file: VocabularyFile) {
    private val idByToken = file.tokens.withIndex().associate { it.value to it.index }
    private val candidatesByFirst = file.tokens.withIndex()
        .filter { it.value.isNotEmpty() && !it.value.startsWith('<') }
        .groupBy({ it.value.first() }, { it })
        .mapValues { (_, values) -> values.sortedByDescending { it.value.length } }

    val size: Int get() = file.tokens.size

    fun encode(text: String, includeBoundaryTokens: Boolean = true): List<Int> {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
        val result = ArrayList<Int>()
        if (includeBoundaryTokens) file.bosId?.let(result::add)
        var offset = 0
        while (offset < normalized.length) {
            val candidate = candidatesByFirst[normalized[offset]]?.firstOrNull {
                normalized.startsWith(it.value, offset)
            }
            if (candidate == null) {
                val codePoint = normalized.codePointAt(offset)
                val token = String(Character.toChars(codePoint))
                result += idByToken[token] ?: file.unknownId
                offset += Character.charCount(codePoint)
            } else {
                result += candidate.index
                offset += candidate.value.length
            }
        }
        if (includeBoundaryTokens) file.eosId?.let(result::add)
        return result
    }

    fun decode(ids: Iterable<Int>, stripSpecial: Boolean = true): String = buildString {
        ids.forEach { id ->
            val token = file.tokens.getOrNull(id) ?: return@forEach
            if (!stripSpecial || !token.startsWith('<')) append(token)
        }
    }.replace("▁", " ").trim()

    fun token(id: Int): String = file.tokens.getOrElse(id) { "" }
}
