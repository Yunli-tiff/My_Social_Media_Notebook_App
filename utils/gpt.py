# utils/gpt.py

import openai
from openai.api_resources.chat.completion import ChatCompletion
import streamlit as st
import os

# 從環境變數讀取 API Key
openai.api_key = os.getenv("OPENAI_API_KEY", None)
if not openai.api_key:
    raise RuntimeError("找不到 OPENAI_API_KEY，請先設定環境變數")

@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def gpt_summarize_and_classify(text: str) -> str:
    prompt = f"""
以下是社群貼文內容：
{text}

請以繁體中文：
1. 生成一段100字內摘要
2. 判斷主題屬於以下類別之一：生活、美食、科技、時事、旅遊、娛樂、學習、其他
請以如下格式回覆：
摘要：...
主題分類：...
"""
    try:
        # 直接用 ChatCompletion 物件呼叫 create，避開舊版代理
        response = ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        return f"GPT 呼叫失敗：{e}"

