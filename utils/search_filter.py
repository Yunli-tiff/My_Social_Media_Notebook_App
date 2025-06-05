import pandas as pd

def filter_notes(df, keyword=None, category=None):
    if keyword:
        df = df[df["原文"].str.contains(keyword, case=False, na=False)]
    if category and category != "全部":
        df = df[df["主題"] == category]
    return df
