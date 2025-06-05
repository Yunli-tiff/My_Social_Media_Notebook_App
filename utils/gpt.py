# utils/gpt.py

import openai
import streamlit as st
import os

# 直接從環境變數讀取，若沒設就顯示錯誤訊息
openai.api_key = os.getenv("OPENAI_API_KEY", None)
if not openai.api_key:
    raise RuntimeError("找不到 OPENAI_API_KEY，請先設定環境變數")

@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def gpt_summarize_and_classify(text: str) -> str:
    """
    接收一段文字，讓 GPT-4 做「100 字內摘要 + 主題分類」。
    回傳格式：
      摘要：xxx
      主題分類：xxx
    """
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
        # 確保使用最新版 ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        # 發生 API 錯誤時，回傳例外訊息，側欄可以顯示
        return f"GPT 呼叫失敗：{e}"
