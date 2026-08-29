from pathlib import Path
import json

REQUIRED=["title","problem","method","dataset","metrics","limitations","reproducibility_notes"]

def main():
    root=Path(__file__).resolve().parents[1]
    review=json.loads((root/"example_review.json").read_text())
    missing=[k for k in REQUIRED if not review.get(k)]
    print("Review complete" if not missing else f"Missing fields: {missing}")
    for k in REQUIRED: print(f"{k}: {review.get(k,'')}")

if __name__ == "__main__": main()
