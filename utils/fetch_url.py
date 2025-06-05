# utils/fetch_url.py

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tempfile

def fetch_page_html(url: str, timeout: int = 10) -> str:
    """
    使用 requests 抓取完整 HTML。
    若需處理防爬蟲（如 Cloudflare），可再考慮 Selenium 或 cloudscraper。
    """
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text

def extract_visible_text(html: str) -> str:
    """
    用 BeautifulSoup 取出 <body> 裡主要可見文字 (去掉 <script>, <style> 等)。
    """
    soup = BeautifulSoup(html, "lxml")
    # 移除 script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    visible_text = soup.get_text(separator="\n")
    # 簡單去掉多餘空行
    lines = [line.strip() for line in visible_text.splitlines()]
    return "\n".join([line for line in lines if line])

def extract_title(html: str) -> str:
    """
    取出 <title> 標籤內容，若沒有則回空字串。
    """
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    return title_tag.get_text().strip() if title_tag else ""

def download_all_images(html: str, base_url: str, save_dir: str) -> list[str]:
    """
    從 <img> 標籤抓出所有 src，下載到 save_dir 資料夾並回傳檔名清單。
    save_dir 必須已存在 (或自行先 os.makedirs(save_dir, exist_ok=True))。
    """
    soup = BeautifulSoup(html, "lxml")
    img_tags = soup.find_all("img")
    saved_files = []
    for idx, img in enumerate(img_tags):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        # 針對相對路徑加上 base_url
        img_url = urljoin(base_url, src)
        try:
            r = requests.get(img_url, stream=True, timeout=10)
            r.raise_for_status()
            ext = os.path.splitext(img_url)[-1].split("?")[0]
            if ext.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
                ext = ".jpg"
            fname = f"image_{idx}{ext}"
            fpath = os.path.join(save_dir, fname)
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            saved_files.append(fpath)
        except Exception:
            continue
    return saved_files

def download_all_audio(html: str, base_url: str, save_dir: str) -> list[str]:
    """
    從 <audio> 或 <source> 標籤抓音訊檔 URL 下載到本地。回傳檔案路徑清單。
    """
    soup = BeautifulSoup(html, "lxml")
    audio_files = []
    # 找 <audio src="…"> 或 <source src="…">
    tags = soup.find_all(["audio", "source"])
    for idx, tag in enumerate(tags):
        src = tag.get("src") or ""
        if not src:
            continue
        audio_url = urljoin(base_url, src)
        try:
            r = requests.get(audio_url, stream=True, timeout=10)
            r.raise_for_status()
            ext = os.path.splitext(audio_url)[-1].split("?")[0]
            if ext.lower() not in [".mp3", ".wav", ".ogg", ".aac", ".flac"]:
                ext = ".mp3"
            fname = f"audio_{idx}{ext}"
            fpath = os.path.join(save_dir, fname)
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            audio_files.append(fpath)
        except Exception:
            continue
    return audio_files
