import openai
import streamlit as st
import os

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def gpt_summarize_and_classify(text: str) -> str:
    prompt = f"""以下是社群貼文內容：
{text}

請以繁體中文：
1. 生成一段100字內摘要
2. 判斷主題屬於以下類別之一：生活、美食、科技、時事、旅遊、娛樂、學習、其他
請以如下格式回覆：
摘要：...
主題分類：...
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
