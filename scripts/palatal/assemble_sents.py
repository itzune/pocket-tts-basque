import json, re, csv, collections, os
eh=json.load(open("eh_examples.json")); eeh=json.load(open("eeh_rows.json")); wiki=json.load(open("wiki_rows.json"))
PAL=re.compile(r"[A-Za-zñ-]*(?:tt|dd|ll)[aeiouáéíóú][A-Za-zñ-]*")
def pal_words(s): return " ".join(sorted(set(w.lower() for w in PAL.findall(s))))
rows=[]
def add(sentence, word, src, url, priority):
    s=re.sub(r"\s+"," ",sentence).strip()
    if len(s)<12 or not PAL.search(s): return
    if "..." in s or s.startswith(("\"", "...")) or re.search(r"\b(entry|headword|contains|I left|EEH onddo)\b", s): return
    rows.append(dict(priority=priority, source=src, word=word, palatal_words=pal_words(s), sentence=s, url=url))
# 1. Euskaltzaindiaren Hiztegia examples + proverbs
for key,e in eh.items():
    w=key.replace("%20"," ")
    url=f"https://www.euskaltzaindia.eus/index.php?option=com_hiztegianbilatu&task=bilaketa&Itemid=1693&lang=eu-ES&non=sarreraBuruaStrict&query={key}"
    for ent in e.get("entries",[]):
        for ex in ent["examples"]: add(ex, w, "Euskaltzaindiaren Hiztegia", url, 1)
# 2. EEH (EHU, published prose) and Elhuyar
for r in eeh:
    add(r["sentence"], r["word"], "EEH (EHU)" if r["source"]=="EEH" else "Elhuyar hiztegia", r["url"], 2 if r["source"]=="EEH" else 3)
# 3. Wikipedia
for r in wiki: add(r["sentence"], r["word"], "eu.wikipedia", "https://eu.wikipedia.org", 4)
# dedupe
seen=set(); out=[]
for r in sorted(rows, key=lambda r:(r["priority"], r["word"])):
    k=r["sentence"].lower()
    if k in seen: continue
    seen.add(k); out.append(r)
dst="resources/palatal/palatal_sentences.tsv"
with open(dst,"w",newline="") as f:
    f.write("# Basque sentences containing palatal words (tt/dd/ll). priority 1 = Euskaltzaindiaren Hiztegia examples, 2 = EEH corpus examples (EHU), 3 = Elhuyar hiztegia, 4 = eu.wikipedia.\n")
    w=csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(out)
c=collections.Counter(r["source"] for r in out); print(len(out), dict(c))
dg=collections.Counter()
for r in out:
    for d in ("dd","tt","ll"):
        if re.search(d+r"[aeiou]", r["sentence"].lower()): dg[d]+=1
print("sentences with", dict(dg))
print("dd sentences by source:", collections.Counter(r["source"] for r in out if re.search(r"dd[aeiou]", r["sentence"].lower())))
