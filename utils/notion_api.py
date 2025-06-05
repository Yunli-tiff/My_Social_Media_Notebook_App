from notion_client import Client

def upload_to_notion(page_id, summary, category, source_text, notion_token):
    notion = Client(auth=notion_token)
    notion.pages.create(
        parent={"database_id": page_id},
        properties={
            "摘要": {"title": [{"text": {"content": summary}}]},
            "分類": {"select": {"name": category}},
        },
        children=[
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"text": [{"type": "text", "text": {"content": source_text[:1900]}}]}
            }
        ]
    )
