# utils/gpt.py

import streamlit as st
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------- 1) 初始化多語系「摘要」模型 (mT5-base) -------------
@st.cache_resource
def load_multilang_summarizer():
    """
    mT5-base 可以做多語言的摘要 (doc -> 100 字左右)
    """
    model_name = "google/mt5-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tok,
        framework="pt",
        # 指定 max_length / min_length 
        # mT5-base 的 input_size 最多 512 tokens，輸出可設定成 100 tokens
        # 也可以在呼叫時傳參調整
    )

# ---------- 2) 初始化多語 zero-shot 分類模型 (XLM-RoBERTa) -------------
@st.cache_resource
def load_multilang_classifier():
    """
    xlm-roberta-large-mnli 支援多語 zero-shot classification
    """
    model_name = "joeddav/xlm-roberta-large-xnli"
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
    )

# ---------- 3) 多語言摘要＋分類 + 關鍵字 (簡單示範) -------------
@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def multilang_summarize_and_classify(text: str) -> Tuple[str, str, List[str]]:
    """
    輸入：任意語言的 text。
    1. 用 mT5-base 做摘要 (100 字左右)。
    2. 用 XLM-RoBERTa 做 zero-shot 分類，分類標籤可自行擴充。
    3. 用最簡單的方式 (數 summary 裡高頻詞) 做「關鍵字擷取」，僅作範例。
    輸出：(summary, top_label, keywords_list)
    """
    # --- 1. 摘要 ---
    summarizer = load_multilang_summarizer()
    try:
        summ = summarizer(
            text,
            max_length=100,   # 輸出控在 100 tokens 左右
            min_length=30,    # 至少 30 tokens
            do_sample=False
        )
        summary_text = summ[0]["summary_text"].strip()
    except Exception as e:
        summary_text = f"❌ 摘要失敗：{e}"

    # --- 2. 多語零樣本分類 ---
    classifier = load_multilang_classifier()
    # 下面這裡列出你想要的「主題標籤」，無論是中文或英文都行
    candidate_labels = ["生活", "美食", "科技", "旅遊", "娛樂", "學習", "商業", "其他"]
    try:
        cls_result = classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False
        )
        top_label = cls_result["labels"][0]
    except Exception as e:
        top_label = f"❌ 分類失敗：{e}"

    # --- 3. 簡易關鍵字擷取 (僅示範中文/英文) ---
    from collections import Counter
    import re

    # 把 summary 清理成單詞 (對中英文都能抓到「詞」)
    words = re.findall(r"\b\w+\b", summary_text.lower())
    # 簡單的中英文混合停用詞列表 (依需求再擴充)
    stopwords = set([
        "的", "了", "在", "是", "我", "也", "和", "就", "不", "有",
        "the", "and", "to", "of", "in", "for", "with", "on", "that"
    ])
    filtered = [w for w in words if w not in stopwords and len(w) > 1]
    most_common = [w for w, _ in Counter(filtered).most_common(5)]
    keywords = most_common

    return summary_text, top_label, keywords

