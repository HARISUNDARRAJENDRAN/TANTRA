package in.tantra.transceiver.protocol

class StablePrefixCommitter(
    private val requiredObservations: Int = 3,
    private val minimumStabilityMs: Long = 450,
) {
    private var lastHypothesis = ""
    private var stablePrefix = ""
    private var stableSinceMs = 0L
    private var observations = 0
    private var committed = ""

    fun update(raw: String, nowMs: Long, final: Boolean = false): String {
        val hypothesis = raw.trim().replace(Regex("\\s+"), " ")
        val common = lcp(lastHypothesis, hypothesis)
        if (common == stablePrefix) observations++ else {
            stablePrefix = common
            stableSinceMs = nowMs
            observations = 1
        }
        lastHypothesis = hypothesis
        var candidate = if (final) hypothesis else stablePrefix
        if (!final) {
            if (observations < requiredObservations && nowMs - stableSinceMs < minimumStabilityMs) return ""
            candidate = candidate.substring(0, safeBoundary(candidate))
        }
        if (candidate.length <= committed.length) return ""
        val emitted = candidate.substring(committed.length).trimStart()
        committed = candidate
        return emitted
    }

    fun reset() {
        lastHypothesis = ""; stablePrefix = ""; stableSinceMs = 0; observations = 0; committed = ""
    }

    private fun lcp(left: String, right: String): String {
        var index = 0
        val limit = minOf(left.length, right.length)
        while (index < limit && left[index] == right[index]) index++
        return left.substring(0, index)
    }

    private fun safeBoundary(value: String): Int {
        for (index in value.length downTo 1) {
            val char = value[index - 1]
            if (char.isWhitespace() || char in ".!?।॥,:;") return index
        }
        return 0
    }
}
