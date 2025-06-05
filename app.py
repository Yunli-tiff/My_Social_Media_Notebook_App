import streamlit as st
import pandas as pd
import os
import time
from utils.ocr import extract_text_from_image
from utils.whisper_asr import transcribe_audio
from utils.gpt import gpt_summarize_and_classify
from utils.search_filter import filter_notes
from utils.markdown_export import export_notes_to_md
from utils.notion_api import upload_to_notion
from utils.dropbox_export import upload_to_dropbox

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 基本設定
st.set_page_config(
    page_title="社群筆記牆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 網頁標題
st.title("📌 社群筆記牆：主題導向的 AI 筆記整理工具")
st.markdown(
    """
    此應用能自動將上傳的圖片（OCR）、音訊（Whisper 語音辨識）、純文字
    內容，透過 GPT 完成「摘要 + 主題分類」，並呈現在可搜尋、可匯出、
    可同步到 Notion / Dropbox 的互動式筆記牆中。
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 側邊欄：操作區
with st.sidebar:
    st.header("🔧 操作區")
    # 1. 上傳檔案
    upload_files = st.file_uploader(
        label="📤 上傳圖片 / 音訊 / 純文字檔案",
        type=["png", "jpg", "jpeg", "mp3", "wav", "txt"],
        accept_multiple_files=True
    )

    # 2. 關鍵字搜尋（後續做篩選）
    keyword = st.text_input("🔍 關鍵字搜尋", placeholder="搜尋筆記原文...")

    # 3. 主題分類下拉：動態生成
    category_options_placeholder = st.empty()

    # 4. Notion API 欄位
    st.markdown("---")
    st.subheader("📒 Notion 同步")
    notion_token = st.text_input(
        "Notion Integration Token", placeholder="輸入你的 Notion API Token", type="password"
    )
    notion_db_id = st.text_input(
        "Notion Database ID", placeholder="輸入目標 Database ID"
    )
    sync_notion_btn = st.button("➡️ 同步到 Notion", key="sync_notion")

    # 5. Dropbox 欄位
    st.markdown("---")
    st.subheader("📁 Dropbox 同步")
    dropbox_token = st.text_input(
        "Dropbox Access Token", placeholder="輸入你的 Dropbox Token", type="password"
    )
    sync_dropbox_btn = st.button("➡️ 備份到 Dropbox", key="sync_dropbox")

    # 6. Markdown 下載按鈕
    st.markdown("---")
    st.subheader("📄 下載匯出")
    export_md_btn = st.button("⬇️ 下載 Markdown", key="export_md")

# ─────────────────────────────────────────────────────────────────────────────
# 主要邏輯：若有檔案上傳，就進行 OCR / Whisper / GPT
note_data = []  # 儲存所有筆記條目

if upload_files:
    st.sidebar.success(f"已上傳 {len(upload_files)} 筆資料，開始處理⋯⋯")
    # 顯示 Spinner
    with st.spinner("📦 處理檔案中 (OCR/ASR/GPT)… 請稍候"):
        for file in upload_files:
            # 1. 讀取檔案內容
            if file.type.startswith("image"):
                # 圖片 OCR
                content = extract_text_from_image(file)
            elif file.type.startswith("audio"):
                # Whisper ASR
                content = transcribe_audio(file)
            else:
                # 純文字檔案 (.txt)
                content = file.read().decode("utf-8")

            # 2. GPT 進行「摘要 + 主題分類」
            summary_output = gpt_summarize_and_classify(content)
            # 回傳格式範例： 
            #  摘要：這是一段 GPT 生成的 100 字以內摘要
            #  主題分類：生活
            lines = [line.strip() for line in summary_output.strip().split("\n") if line.strip()]
            if len(lines) >= 2:
                summary = lines[0].replace("摘要：", "").strip()
                category = lines[1].replace("主題分類：", "").strip()
            else:
                # 抓不到格式則放到「其他」
                summary = summary_output[:100]
                category = "其他"

            note_data.append({
                "檔名": file.name,
                "主題": category,
                "摘要": summary,
                "原文": content
            })
        time.sleep(0.5)  # 微幅暫停，避免 Spinner 一閃而過
    st.sidebar.success("✅ 所有檔案已處理完成！")

# ─────────────────────────────────────────────────────────────────────────────
# 若已經產生 note_data，將它轉成 DataFrame
if note_data:
    notes_df = pd.DataFrame(note_data)

    # 1. 先更新「主題分類」下拉選單選項：先放「全部」再依序放 actual categories
    category_list = ["全部"] + sorted(notes_df["主題"].unique().tolist())
    category = category_options_placeholder.selectbox(
        "🗂️ 選擇主題分類",
        category_list,
        index=0
    )

    # 2. 篩選：先篩「關鍵字」，再篩「主題分類」
    filtered_df = filter_notes(notes_df, keyword=keyword, category=category)

    # 3. 左側顯示統計資訊
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 總共上傳檔案數", len(upload_files))
    col2.metric("📑 篩選後筆記數", len(filtered_df))
    distinct_topics = filtered_df["主題"].nunique()
    col3.metric("📂 篩選後主題數", distinct_topics)

    st.markdown("---")

    # 4. 將篩選後的筆記按「主題」分組，排成兩欄展示
    grouped = filtered_df.groupby("主題")

    for topic, group in grouped:
        st.subheader(f"📂 {topic} ({len(group)})")
        # 兩欄配置：若只有一筆，則塞到左邊
        left_col, right_col = st.columns(2)
        for idx, row in group.iterrows():
            with (left_col if (idx % 2 == 0) else right_col).expander(f"📎 {row['檔名']}"):
                st.markdown(f"**摘要：** {row['摘要']}")
                st.markdown(f"**原文內容：**\n{row['原文'][:1000]}{'...' if len(row['原文'])>1000 else ''}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. 處理：按鈕回呼
    #    A. Markdown 匯出
    if export_md_btn:
        export_path = export_notes_to_md(filtered_df.to_dict("records"), path="notes_export.md")
        with open(export_path, "rb") as f:
            st.sidebar.download_button(
                label="⬇️ 下載 notes_export.md",
                data=f,
                file_name="notes_export.md",
                mime="text/markdown"
            )

    #    B. 同步到 Notion
    if sync_notion_btn:
        if not notion_token or not notion_db_id:
            st.sidebar.error("⚠️ 請先填寫 Notion Token 與 Database ID！")
        else:
            with st.spinner("🔄 同步中…請稍候"):
                success_count = 0
                for _, row in filtered_df.iterrows():
                    try:
                        upload_to_notion(
                            page_id=notion_db_id,
                            summary=row["摘要"],
                            category=row["主題"],
                            source_text=row["原文"],
                            notion_token=notion_token
                        )
                        success_count += 1
                    except Exception as e:
                        st.sidebar.error(f"同步失敗：{row['檔名']} － {e}")
                time.sleep(0.5)
            st.sidebar.success(f"✅ 已成功同步 {success_count} 筆到 Notion！")

    #    C. 備份到 Dropbox
    if sync_dropbox_btn:
        if not dropbox_token:
            st.sidebar.error("⚠️ 請先填寫 Dropbox Token！")
        else:
            # 先建立臨時 Markdown，再上傳
            tmp_md = export_notes_to_md(filtered_df.to_dict("records"), path="notes_backup.md")
            with st.spinner("☁️ 備份中…請稍候"):
                try:
                    # Dropbox 路徑：直接放在根目錄下
                    dropbox_path = f"/notes_backup_{int(time.time())}.md"
                    upload_to_dropbox(
                        token=dropbox_token,
                        local_file_path=tmp_md,
                        dropbox_dest_path=dropbox_path
                    )
                    st.sidebar.success(f"✅ 已備份至 Dropbox：{dropbox_path}")
                except Exception as e:
                    st.sidebar.error(f"Dropbox 備份失敗：{e}")
                    raise

else:
    # 若尚未上傳任何檔案，顯示提示
    st.info("請先在左側「操作區」上傳圖片 / 音訊 / 文字檔，系統才會自動產生筆記。")
