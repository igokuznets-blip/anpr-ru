"""Правила номерных знаков РФ (тип 1: А123ВС77 / А123ВС777) и нормализация результата OCR."""
import re

RU_LETTERS = "АВЕКМНОРСТУХ"                 # 12 букв, допустимых на номерах РФ
LAT_LETTERS = "ABEKMHOPCTYX"                # их латинские двойники (алфавит датасетов)
OCR_ALPHABET = "0123456789" + LAT_LETTERS   # алфавит CRNN-модели

LAT2CYR = str.maketrans(LAT_LETTERS, RU_LETTERS)
TO_DIGIT = {"О": "0", "O": "0", "В": "8", "B": "8", "З": "3", "Б": "6",
            "Т": "7", "I": "1", "L": "1", "S": "5", "D": "0", "Q": "0"}
TO_LETTER = {"0": "О", "8": "В", "3": "З", "4": "А", "7": "Т", "6": "В", "1": "Т"}
PLATE_RE = re.compile(rf"^[{RU_LETTERS}]\d{{3}}[{RU_LETTERS}]{{2}}\d{{2,3}}$")
MASK = "LDDDLLDDD"                          # Буква Цифра×3 Буква×2 Регион(2–3 цифры)


def normalize_plate(raw: str) -> str | None:
    """Привести строку OCR к номеру РФ в кириллице или вернуть None."""
    s = re.sub(r"[^0-9A-ZА-ЯЁ]", "", (raw or "").upper()).translate(LAT2CYR)
    if not 8 <= len(s) <= 9:
        return None
    out = []
    for ch, kind in zip(s, MASK):
        if kind == "D" and not ch.isdigit():
            ch = TO_DIGIT.get(ch, ch)
        elif kind == "L" and ch not in RU_LETTERS:
            ch = TO_LETTER.get(ch, ch)
        out.append(ch)
    s = "".join(out)
    return s if PLATE_RE.match(s) else None


def pretty(p: str) -> str:
    """А123ВС77 -> 'А123ВС 77'."""
    return f"{p[:6]} {p[6:]}"


def to_latin(p: str) -> str:
    """Кириллический номер -> латиница (формат имён файлов датасета)."""
    return p.translate(str.maketrans(RU_LETTERS, LAT_LETTERS))
