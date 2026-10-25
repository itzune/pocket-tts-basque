import pyarrow.parquet as pq, re, collections, glob, json, sys
PAL = re.compile(r"\b[\wñ-]*(?:^|[aeiouáéíóú])(?:tt|dd|ll)[aeiouáéíóú][\wñ-]*\b|\b(?:tt|dd|ll)[aeiouáéíóú][\wñ-]*\b", re.I)
WORD = re.compile(r"[a-zñáéíóúü-]+(?:tt|dd|ll)[a-zñáéíóúü-]*", re.I)
freq = collections.Counter(); sents = collections.defaultdict(list)
SPLIT = re.compile(r"(?<=[.!?])\s+")
n=0
for f in glob.glob("wiki/**/*.parquet", recursive=True):
    for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text"]):
        for t in b.column("text").to_pylist():
            for s in SPLIT.split(t.replace("\n", " ")):
                s=s.strip()
                if not (30 <= len(s) <= 220): continue
                hits=set()
                for w in WORD.findall(s):
                    wl=w.lower()
                    if not re.search(r"(tt|dd|ll)[aeiouáéíóú]", wl): continue
                    hits.add(wl)
                for wl in hits:
                    freq[wl]+=1
                    if len(sents[wl])<40: sents[wl].append(s)
            n+=1
print("docs", n, "types", len(freq), file=sys.stderr)
json.dump({"freq":freq.most_common(), "sents":sents}, open("wiki_pal.json","w"), ensure_ascii=False)
