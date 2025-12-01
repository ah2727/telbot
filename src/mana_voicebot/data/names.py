from __future__ import annotations

import re
from typing import List

# Minimal seed; extend as you like
IRANIAN_DEFAULT_NAMES: List[str] = [
    "محمد",
    "محمدرضا",
    "علی",
    "زهرا",
    "فاطمه",
    "حسین",
    "رضا",
    "مریم",
    "سارا",
    "نیما",
]


def normalize_persian_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()
    s = s.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    s = re.sub(r"\s+", " ", s)
    return s


def build_name_prompt() -> str:
    base = (
        "این تماس برای رزرو نوبت است. نام و نام‌خانوادگی فارسی مراجعه‌کننده را دقیق بنویس. "
        "اگر در فایل فقط موسیقی، نویز یا صداهای مبهم شنیدی و گفتار واضح فارسی وجود نداشت، "
        "خروجی را خالی بگذار و هیچ متنی تولید نکن."
    )
    if not IRANIAN_DEFAULT_NAMES:
        return base
    tail = "، ".join(IRANIAN_DEFAULT_NAMES[:20])
    return base + f" چند نام رایج: {tail}."
