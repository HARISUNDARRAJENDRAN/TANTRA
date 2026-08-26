package in.tantra.transceiver.model

enum class Language(val wireId: Int, val bcp47: String, val displayName: String) {
    UNKNOWN(0, "und", "Auto"),
    HINDI(1, "hi-IN", "Hindi"),
    GUJARATI(2, "gu-IN", "Gujarati"),
    MARATHI(3, "mr-IN", "Marathi"),
    KANNADA(4, "kn-IN", "Kannada"),
    MALAYALAM(5, "ml-IN", "Malayalam"),
    TAMIL(6, "ta-IN", "Tamil"),
    TELUGU(7, "te-IN", "Telugu"),
    ODIA(8, "or-IN", "Odia"),
    BENGALI(9, "bn-IN", "Bengali"),
    ENGLISH(10, "en-IN", "English");

    companion object {
        fun fromWireId(value: Int): Language = entries.firstOrNull { it.wireId == value } ?: UNKNOWN
        fun fromCode(code: String): Language = entries.firstOrNull {
            it.bcp47.substringBefore('-').equals(code.substringBefore('-'), ignoreCase = true)
        } ?: UNKNOWN
    }
}
