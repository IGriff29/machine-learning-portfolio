from pathlib import Path
from collections import Counter
import re

def main():
    path=Path(__file__).resolve().parents[1]/"data"/"sample.log"
    levels=Counter(); errors=Counter()
    for line in path.read_text().splitlines():
        m=re.match(r"\S+ \S+ (INFO|WARNING|ERROR) (.*)", line)
        if not m: continue
        level,msg=m.groups(); levels[level]+=1
        if level=="ERROR": errors[msg]+=1
    print("Severity counts:", dict(levels))
    print("Repeated errors:")
    for msg,count in errors.most_common(): print(count, msg)

if __name__ == "__main__": main()
