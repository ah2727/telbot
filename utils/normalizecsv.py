import pandas as pd
import re

def normalize_persian_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()
    s = s.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    s = re.sub(r"\s+", " ", s)
    return s

df = pd.read_csv("----.csv")  # همون نمونه‌ای که دادی
df["first_name_clean"] = df["first_name"].map(normalize_persian_name)

df_clean = (
    df[["first_name_clean"]]
    .drop_duplicates()
    .rename(columns={"first_name_clean": "first_name"})
)

df_clean.to_csv("iranian_names_clean.csv", index=False, encoding="utf-8")
df_clean["first_name"].to_csv("iranian_names.txt", index=False, header=False, encoding="utf-8")