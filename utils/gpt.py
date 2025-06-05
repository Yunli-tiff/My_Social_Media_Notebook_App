# utils/gpt.py

import os
import streamlit as st
from transformers import pipeline, Pipeline
from typing import Tuple, List

# -------------- 初始化兩個 pipeline --------------
# 1. 摘要模型：facebook/bart-large-cnn
#    會自動從 Hugging Face 下載模型並緩存在本機
@st.experimental_singleton
def load_summarization_model() -> Pipeline:
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        framework="pt",  # 指定 PyTorch
    )

# 2. 零樣本分類模型：facebook/bart-large-mnli
#    用於在多個主題中做分類（zero-shot）
@st.experimental_singleton
def load_zero_shot_model() -> Pipeline:
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli",
        framework="pt",
    )

# -------------- 主功能：摘要 + 分類 --------------
@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def gpt_summarize_and_classify(text: str) -> Tuple[str, str]:
    """
    使用 Hugging Face Transformers 做：
      1. 摘要（summarization）
      2. 零樣本分類（zero-shot classification）

    輸入：一段長文本 text
    輸出：tuple(摘要, 分類結果)
    """

    # 先做摘要
    summarizer = load_summarization_model()
    # bart-large-cnn 的預設 max_length 1024；我們限制輸入長度與摘要長度
    # 如果輸入超過 1024 tokens，可自行調整或切分
    try:
        summary_list = summarizer(
            text,
            max_length=100,   # 控制輸出約 100 字左右
            min_length=30,    # 最少 30 字
            do_sample=False   # 禁止隨機取樣（採 beam search）
        )
        summary = summary_list[0]["summary_text"].strip()
    except Exception as e:
        summary = f"❌ 摘要失敗：{e}"

    # 再做分類
    classifier = load_zero_shot_model()
    candidate_labels: List[str] = [
        "生活", "美食", "科技", "時事", "旅遊", "娛樂", "學習", "其他"
    ]
    try:
        cls_result = classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False  # 單一最可能分類
        )
        # cls_result 會包含 keys: 'labels'（排序後的分類）、'scores'
        top_label = cls_result["labels"][0]
    except Exception as e:
        top_label = f"❌ 分類失敗：{e}"

    return summary, top_label
