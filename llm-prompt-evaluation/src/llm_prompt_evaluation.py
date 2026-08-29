from pathlib import Path
import csv

REQUIRED = ["summary", "risk", "next step"]

def score(text):
    lower=text.lower()
    completeness=sum(term in lower for term in REQUIRED)/len(REQUIRED)
    concise=max(0.0, 1 - max(0, len(text.split())-90)/90)
    return round(.8*completeness + .2*concise, 3)

def main():
    root=Path(__file__).resolve().parents[1]
    prompts=sorted((root/"prompts").glob("*.txt"))
    sample={
      "prompt_v1.txt":"Summary: service latency increased. Next step: inspect logs.",
      "prompt_v2.txt":"Summary: service latency increased. Risk: delayed responses. Next step: inspect logs and compare recent deployments.",
      "prompt_v3.txt":"Summary: latency rose after a deployment. Risk: degraded user experience. Next step: compare deployment changes, error logs, and rollback criteria.",
    }
    rows=[]
    for p in prompts:
        rows.append({"prompt":p.name,"score":score(sample[p.name]),"prompt_words":len(p.read_text().split())})
    out=root/"results"/"prompt_comparison.csv"; out.parent.mkdir(exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(*rows, sep="\n")

if __name__ == "__main__": main()
