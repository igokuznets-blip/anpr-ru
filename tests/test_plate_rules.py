import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "anpr"))
from plate_rules import normalize_plate, pretty, to_latin  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("А123ВС77", "А123ВС77"),
    ("A123BC77", "А123ВС77"),          # латиница -> кириллица
    ("a123bc777", "А123ВС777"),        # регистр, 3-значный регион
    ("А 123 ВС 77", "А123ВС77"),       # пробелы
    ("O123OO99", "О123ОО99"),
    ("0123ОО99", "О123ОО99"),          # 0 на месте буквы -> О
    ("АО23ВС77", "А023ВС77"),          # О на месте цифры -> 0
    ("В8Б5ТТ197", "В865ТТ197"),
    ("Х340НЕ750", "Х340НЕ750"),
])
def test_valid(raw, expected):
    assert normalize_plate(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "A1Z3BC77", "123456", "Ж123ВС77", "А123ВС7777", "АБВГДЕЖЗ"])
def test_invalid(raw):
    assert normalize_plate(raw) is None


def test_pretty_and_latin():
    assert pretty("А123ВС77") == "А123ВС 77"
    assert to_latin("А123ВС777") == "A123BC777"
