from pathlib import Path
import json

def main():
    root=Path(__file__).resolve().parents[1]
    index=json.loads((root/"prompt_index.json").read_text())
    active=[x for x in index if x["status"]=="active"]
    for item in active:
        text=(root/item["file"]).read_text().strip()
        print(f"{item['name']} v{item['version']}: {text}")

if __name__ == "__main__": main()
