from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "protocol/python"))
sys.path.insert(0, str(ROOT / "ml"))

from tantra_protocol.codec import TextDelta, encode_text_delta
from tantra_ml.text.vocabulary import build_vocabulary

PHRASES = {
    "hi": "कृपया तुरंत चिकित्सा सहायता भेजें",
    "gu": "કૃપા કરીને તરત તબીબી સહાય મોકલો",
    "mr": "कृपया तातडीने वैद्यकीय मदत पाठवा",
    "kn": "ದಯವಿಟ್ಟು ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಕಳುಹಿಸಿ",
    "ml": "ദയവായി ഉടൻ വൈദ്യസഹായം അയയ്ക്കുക",
    "ta": "தயவுசெய்து உடனடியாக மருத்துவ உதவி அனுப்பவும்",
    "te": "దయచేసి వెంటనే వైద్య సహాయం పంపండి",
    "or": "ଦୟାକରି ତୁରନ୍ତ ଚିକିତ୍ସା ସହାୟତା ପଠାନ୍ତୁ",
    "bn": "অনুগ্রহ করে অবিলম্বে চিকিৎসা সহায়তা পাঠান",
    "en": "please send medical assistance immediately",
}


def main() -> None:
    vocabulary = build_vocabulary(PHRASES.values())
    rows = []
    for language, text in PHRASES.items():
        utf8 = len(text.encode("utf-8"))
        tokens = vocabulary.encode(text)
        token_payload = len(encode_text_delta(TextDelta(0, 0, tuple(tokens))))
        rows.append({
            "language": language, "characters": len(text), "tokens": len(tokens),
            "utf8_payload_bytes": utf8, "token_delta_bytes": token_payload,
            "payload_reduction_percent": round((1 - token_payload / utf8) * 100, 2),
            "wire_bytes_with_frame": token_payload + 27,
        })
    result = {
        "schema_version": 1,
        "note": "Small phrase benchmark; vocabulary was built from this phrase set and is not a production entropy estimate.",
        "mean_payload_reduction_percent": round(statistics.mean(row["payload_reduction_percent"] for row in rows), 2),
        "languages": rows,
    }
    output = ROOT / "benchmarks/protocol-phrase-size.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
