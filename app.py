import streamlit as st
import pandas as pd
import os
import time
import tempfile
import re
from utils.fetch_url import (
    fetch_page_html,
    extract_visible_text,
    extract_title,
    download_all_images,
    download_all_audio,
)
from utils.ocr import extract_text_from_image
from utils.whisper_asr import transcribe_audio
from utils.gpt import multilang_summarize_and_classify  # 可以改成你自己的多語摘要函式
from utils.search_filter import filter_notes
from utils.markdown_export import export_notes_to_md
from utils.notion_api import upload_to_notion
from utils.dropbox_export import upload_to_dropbox

# ──────────── 輔助函式：從文字中擷取所有 URL ────────────
URL_REGEX = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}"
    r"\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

def extract_urls_from_text(text: str) -> list[str]:
    """
    從一大段文字裡，找出所有符合 URL_REGEX 的連結。
    """
    return re.findall(URL_REGEX, text)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 基本設定
st.set_page_config(
    page_title="社群筆記牆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 網頁標題
st.title("📌 InstaNote：你的專屬AI筆記牆")
st.markdown(
    """
    這是一個主題導向的AI互動式筆記牆，能讀取網頁或上傳的多種內容，進行自動化摘要、分類和儲存，
    幫你省下大量資料整理的時間！系統支援多語系模型，以及篩選、匯出、同步到 Notion / Dropbox等功能～
    你只需要貼入一個或多個網址，或預先上傳下載好的圖片／音訊／文字檔，
    就可以一鍵體驗以上功能。InstaNote絕對是懶人、J人的福音！
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 側邊欄：操作區
with st.sidebar:
    st.header("🔧 操作區")

    # 1. 上傳檔案（圖片/音訊/文字）
    st.subheader("📤 上傳圖片 / 音訊 / 純文字檔案")
    upload_files = st.file_uploader(
        label="點擊上傳...", 
        type=["png", "jpg", "jpeg", "mp3", "wav", "txt"], 
        accept_multiple_files=True
    )

    # 2. 貼入一或多個網址
    st.markdown("---")
    st.subheader("🌐 貼入一或多個網址 (一行一個或空格分隔)")
    paste_urls = st.text_area(
        label="請直接貼上 URL (http:// 或 https:// 開頭)",
        placeholder="https://example.com/page1  https://example.com/page2\n或換行貼多個網址...",
        height=100
    )
    process_urls_btn = st.button("➡️ 處理貼入的網址", key="process_urls")

    # 3. 關鍵字搜尋
    st.markdown("---")
    keyword = st.text_input("🔍 關鍵字搜尋", placeholder="搜尋筆記內容...")

    # 4. 主題分類下拉：動態生成，之後會依 note_data 填入
    category_options_placeholder = st.empty()

    # 5. Notion 同步
    st.markdown("---")
    st.subheader("📒 Notion 同步")
    notion_token = st.text_input(
        "Notion Integration Token", 
        placeholder="輸入你的 Notion API Token", 
        type="password"
    )
    notion_db_id = st.text_input(
        "Notion Database ID", 
        placeholder="輸入目標 Database ID"
    )
    sync_notion_btn = st.button("➡️ 同步到 Notion", key="sync_notion")

    # 6. Dropbox 同步
    st.markdown("---")
    st.subheader("📁 Dropbox 同步")
    dropbox_token = st.text_input(
        "Dropbox Access Token", 
        placeholder="輸入你的 Dropbox Token", 
        type="password"
    )
    sync_dropbox_btn = st.button("➡️ 備份到 Dropbox", key="sync_dropbox")

    # 7. Markdown 下載
    st.markdown("---")
    st.subheader("📄 Markdown 匯出")
    export_md_btn = st.button("⬇️ 下載 Markdown", key="export_md")

# ─────────────────────────────────────────────────────────────────────────────
# 主要邏輯：note_data 同時儲存「上傳檔案」及「貼入網址」的結果
note_data = []

# ———— 1) 處理：使用者上傳檔案 ————
if upload_files:
    st.sidebar.success(f"已上傳 {len(upload_files)} 筆檔案，開始處理⋯⋯")
    with st.spinner("📦 進行 OCR/ASR + 多語摘要分類…"):
        for file in upload_files:
            # 如果上傳的是 .txt，且裡面可能有多個 URL，先抽出來批次處理
            if file.name.lower().endswith(".txt"):
                raw_text = file.read().decode("utf-8")
                urls = extract_urls_from_text(raw_text)

                if urls:
                    # 若抓到至少一個 URL，就按每個 URL 處理
                    for url in urls:
                        try:
                            html = fetch_page_html(url)
                            title = extract_title(html)
                            text_content = extract_visible_text(html)

                            # 建立臨時資料夾，下載多媒體
                            tmp_dir = os.path.join(tempfile.gettempdir(), "url_fetch")
                            os.makedirs(tmp_dir, exist_ok=True)
                            imgs = download_all_images(html, url, tmp_dir)
                            audios = download_all_audio(html, url, tmp_dir)

                            # 圖片做 OCR、音訊做 ASR
                            for img_path in imgs:
                                text_content += "\n" + extract_text_from_image(open(img_path, "rb"))
                            for audio_path in audios:
                                text_content += "\n" + transcribe_audio(open(audio_path, "rb"))

                            summary, category, keywords = multilang_summarize_and_classify(text_content)

                            record = {
                                "type": "url_batch",
                                "source": url,
                                "url": url,
                                "title": title,
                                "content": text_content,
                                "summary": summary,
                                "category": category,
                                "keywords": keywords,
                                "media": imgs + audios
                            }
                            note_data.append(record)
                        except Exception as e:
                            st.sidebar.error(f"❌ 無法讀取或處理網址：{url}\n請先下載內容再上傳檔案 ({e})")
                else:
                    # 純文字檔但無 URL，直接當成一筆「文字內容」跑摘要
                    content = raw_text
                    summary, category, keywords = multilang_summarize_and_classify(content)
                    record = {
                        "type": "text",
                        "source": file.name,
                        "url": "",
                        "title": file.name,
                        "content": content,
                        "summary": summary,
                        "category": category,
                        "keywords": keywords,
                        "media": []
                    }
                    note_data.append(record)
                continue  # 處理完這個 .txt 檔後，繼續下一個上傳檔案

            # 非 .txt：圖片、音訊、或單純文字檔 (其他副檔名)
            if file.type.startswith("image"):
                content = extract_text_from_image(file)
                media_paths = []
            elif file.type.startswith("audio"):
                content = transcribe_audio(file)
                media_paths = []
            else:
                content = file.read().decode("utf-8")
                media_paths = []

            summary, category, keywords = multilang_summarize_and_classify(content)
            record = {
                "type": "file",
                "source": file.name,
                "url": "",
                "title": file.name,
                "content": content,
                "summary": summary,
                "category": category,
                "keywords": keywords,
                "media": media_paths
            }
            note_data.append(record)

        time.sleep(0.5)
    st.sidebar.success("✅ 上傳檔案處理完成！")

# ———— 2) 處理：使用者直接貼入網址 ————
if process_urls_btn and paste_urls:
    # 從 paste_urls 多行文字裡抽出所有 URL
    urls = extract_urls_from_text(paste_urls)
    if not urls:
        st.sidebar.error("❌ 這段文字中找不到有效的 URL，請確認格式或直接下載後上傳檔案。")
    else:
        st.sidebar.success(f"共偵測到 {len(urls)} 個網址，開始批次擷取⋯⋯")
        with st.spinner("🌐 擷取並處理貼入的網址…"):
            for url in urls:
                try:
                    html = fetch_page_html(url)
                    title = extract_title(html)
                    text_content = extract_visible_text(html)

                    tmp_dir = os.path.join(tempfile.gettempdir(), "url_fetch")
                    os.makedirs(tmp_dir, exist_ok=True)
                    imgs = download_all_images(html, url, tmp_dir)
                    audios = download_all_audio(html, url, tmp_dir)

                    for img_path in imgs:
                        text_content += "\n" + extract_text_from_image(open(img_path, "rb"))
                    for audio_path in audios:
                        text_content += "\n" + transcribe_audio(open(audio_path, "rb"))

                    summary, category, keywords = multilang_summarize_and_classify(text_content)

                    record = {
                        "type": "url",
                        "source": url,
                        "url": url,
                        "title": title,
                        "content": text_content,
                        "summary": summary,
                        "category": category,
                        "keywords": keywords,
                        "media": imgs + audios
                    }
                    note_data.append(record)
                except Exception as e:
                    st.sidebar.error(f"❌ 網址 {url} 處理失敗：{e}\n請先下載檔案再上傳。")
            time.sleep(0.5)
        st.sidebar.success("✅ 貼入網址處理完成！")

# ─────────────────────────────────────────────────────────────────────────────
# 若 note_data 有內容，將它轉成 DataFrame 並顯示
if note_data:
    notes_df = pd.DataFrame(note_data)

    # 1. 更新「主題分類」下拉：先放「全部」再依序放實際 categories
    category_list = ["全部"] + sorted(notes_df["category"].unique().tolist())
    category = category_options_placeholder.selectbox(
        "🗂️ 選擇主題分類",
        category_list,
        index=0
    )

    # 2. 篩選：先按關鍵字，再按主題分類
    filtered_df = filter_notes(
        notes_df.rename(columns={"category": "主題", "content": "原文", "summary": "摘要"}),
        keyword=keyword,
        category=category
    )

    # 3. 左側顯示統計資訊
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 總筆記數", len(notes_df))
    col2.metric("📑 篩選後筆記數", len(filtered_df))
    distinct_topics = filtered_df["主題"].nunique()
    col3.metric("📂 篩選後主題數", distinct_topics)

    st.markdown("---")

    # 4. 按「主題」分組，兩欄顯示每筆筆記
    grouped = filtered_df.groupby("主題")
    for topic, group in grouped:
        st.subheader(f"📂 {topic} ({len(group)})")
        left_col, right_col = st.columns(2)
        for idx, row in group.iterrows():
            label = row["title"] or row["source"]
            with (left_col if (idx % 2 == 0) else right_col).expander(f"📎 {label}"):
                st.markdown(f"**摘要：** {row['摘要']}")
                st.markdown(f"**原文內容：**\n{row['原文'][:1000]}{'...' if len(row['原文'])>1000 else ''}")

    # 5. 按鈕回呼：Markdown 匯出、Notion 同步、Dropbox
    if export_md_btn:
        export_path = export_notes_to_md(
            filtered_df.rename(columns={
                "category": "主題",
                "content": "原文",
                "summary": "摘要",
                "keywords": "關鍵字",
                "url": "網址",
                "title": "標題"
            }).to_dict("records"),
            path="notes_export.md"
        )
        with open(export_path, "rb") as f:
            st.sidebar.download_button(
                label="⬇️ 下載 notes_export.md",
                data=f,
                file_name="notes_export.md",
                mime="text/markdown"
            )

    if sync_notion_btn:
        if not notion_token or not notion_db_id:
            st.sidebar.error("⚠️ 請先填寫 Notion Token 與 Database ID！")
        else:
            with st.spinner("🔄 同步到 Notion 中…"):
                success_count = 0
                for _, row in filtered_df.iterrows():
                    try:
                        upload_to_notion(
                            page_id=notion_db_id,
                            summary=row["摘要"],
                            category=row["主題"],
                            source_text=row["原文"],
                            notion_token=notion_token,
                            url=row["url"],
                            title=row["title"],
                            keywords=row["keywords"]
                        )
                        success_count += 1
                    except Exception as e:
                        st.sidebar.error(f"同步失敗：{row['source']} － {e}")
                time.sleep(0.5)
            st.sidebar.success(f"✅ 已成功同步 {success_count} 筆到 Notion！")

    if sync_dropbox_btn:
        if not dropbox_token:
            st.sidebar.error("⚠️ 請先填寫 Dropbox Token！")
        else:
            tmp_md = export_notes_to_md(
                filtered_df.rename(columns={
                    "category": "主題",
                    "content": "原文",
                    "summary": "摘要",
                    "keywords": "關鍵字",
                    "url": "網址",
                    "title": "標題"
                }).to_dict("records"),
                path="notes_backup.md"
            )
            with st.spinner("☁️ 備份到 Dropbox 中…"):
                try:
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
    st.info("請先在左側「操作區」上傳檔案、或貼入網址，系統才會自動產生筆記。")
