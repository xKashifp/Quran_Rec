import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surah_meta import SURAHS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
XML = DATA / "quran-uthmani.xml"
TANZIL_URL = "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=xml&agree=true"


def main() -> int:
    DATA.mkdir(exist_ok=True)
    if not XML.exists():
        print("downloading Tanzil Uthmani text ...")
        urllib.request.urlretrieve(TANZIL_URL, XML)

    surahs = [{"number": n, "name_ar": ar, "name_en": en, "translation": tr, "ayah_count": c}
              for n, ar, en, tr, c, _ in SURAHS]

    ayahs = []
    for sura in ET.parse(XML).getroot().iter("sura"):
        s = int(sura.get("index"))
        for aya in sura.iter("aya"):
            item = {"surah": s, "ayah": int(aya.get("index")), "text": aya.get("text")}
            if aya.get("bismillah"):
                item["bismillah"] = aya.get("bismillah")
            ayahs.append(item)

    assert len(ayahs) == 6236, len(ayahs)
    out = {
        "source": "Tanzil Quran Text (Uthmani, v1.1), tanzil.net, CC BY 3.0. Text unmodified.",
        "surahs": surahs,
        "ayahs": ayahs,
    }
    (DATA / "quran_text.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote data/quran_text.json: {len(surahs)} surahs, {len(ayahs)} ayahs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
