def export_notes_to_md(notes, path="notes.md"):
    with open(path, "w", encoding="utf-8") as f:
        for note in notes:
            f.write(f"# {note['檔名']}\n")
            f.write(f"**主題：** {note['主題']}\n\n")
            f.write(f"**摘要：** {note['摘要']}\n\n")
            f.write("**原文內容：**\n")
            f.write(f"{note['原文']}\n\n---\n")
    return path
