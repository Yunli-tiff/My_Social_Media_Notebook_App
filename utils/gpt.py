# utils/gpt.py

import streamlit as st
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------------------------------------------------------------
# 1) 多語言摘要：使用 mT5-base（支援中、英、日、韓……等 50+ 種語言）
# ------------------------------------------------------------------------
@st.cache_resource
def load_multilang_summarizer():
    """
    載入 mT5-base 模型，做多語系的摘要 (最多 100 tokens 左右)。
    """
    model_name = "google/mt5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        # max_length、min_length 可以在呼叫時再進行微調
    )

# ------------------------------------------------------------------------
# 2) 多語零樣本分類：使用 XLM-RoBERTa-large-xnli
# ------------------------------------------------------------------------
@st.cache_resource
def load_multilang_classifier():
    """
    載入 xlm-roberta-large-xnli，做多語 zero-shot 分類。
    """
    model_name = "joeddav/xlm-roberta-large-xnli"
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
    )

# ------------------------------------------------------------------------
# 3) 多語摘要＋分類 + 簡易關鍵字擷取
# ------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def multilang_summarize_and_classify(text: str) -> Tuple[str, str, List[str]]:
    """
    輸入任意語言的長文本 (text)，
    1) 用 mT5-base 做摘要 (大約 100 tokens 左右)。
    2) 用 xlm-roberta-large-xnli 做 zero-shot 分類 (多語支援)。
    3) 用最簡易的方式 (統計摘要裡最常見的詞) 做關鍵字擷取，僅作示範。

    回傳：
      (summary: str, category: str, keywords: List[str])
    """
    # --- 1. 摘要 ---
    summarizer = load_multilang_summarizer()
    try:
        # max_length=100：輸出在 100 tokens 左右；min_length=30：至少 30 tokens
        result = summarizer(
            text,
            max_length=100,
            min_length=30,
            do_sample=False
        )
        summary_text = result[0]["summary_text"].strip()
    except Exception as e:
        summary_text = f"❌ 摘要失敗：{e}"

    # --- 2. 分類 ---
    classifier = load_multilang_classifier()
    # 你可以自行增減下面這些「主題標籤」
    candidate_labels = ["生活", "美食", "科技", "旅遊", "娛樂", "學習", "商業", "其他"]
    try:
        cls_result = classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False  # 單選最可能的一個分類
        )
        top_label = cls_result["labels"][0]
    except Exception as e:
        top_label = f"❌ 分類失敗：{e}"

    # --- 3. 簡易關鍵字擷取（示範用，僅取 summary 裡最常見的 5 個詞） ---
    from collections import Counter
    import re

    # 把 summary_text 轉成小寫，並切出所有「字詞」
    words = re.findall(r"\b\w+\b", summary_text.lower())
    # 這邊給出一個中英文混合的簡易停用詞列表，實務可自行擴充
    stopwords = set([
        "的", "了", "在", "是", "我", "也", "和", "就", "不", "有",
        "the", "and", "to", "of", "in", "for", "with", "on", "that"
    ])
    filtered = [w for w in words if w not in stopwords and len(w) > 1]
    most_common = [w for w, _ in Counter(filtered).most_common(5)]
    keywords = most_common

    return summary_text, top_label, keywords
