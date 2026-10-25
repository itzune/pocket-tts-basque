import re, html, sys, json, glob
def clean(s): return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>','',s))).strip()
out={}
for f in sorted(glob.glob("eh/*.html")):
    w=f[3:-5]; h=open(f,encoding="utf-8",errors="ignore").read()
    entries=[]
    for m in re.finditer(r'<div xmlns="[^"]*" class="sarrera" id="(\d+)">(.*?)</div>\s*</div>\s*<div class="row eh-botonera">', h, re.S):
        seg=m.group(2)
        head=clean(re.search(r'class="sarrera-burua[^"]*">(.*?)</div>', seg, re.S).group(1))
        defs=[clean(x) for x in re.findall(r'<span class="def">(.*?)</span>', seg, re.S)]
        marks=[clean(x) for x in re.findall(r"class='laburdura-etzana'[^>]*>(.*?)</span>", seg, re.S)]
        ex=[clean(x) for x in re.findall(r'<span class="dicteg">(.*?)</span>', seg, re.S)]
        entries.append({"head":head,"marks":sorted(set(marks)),"defs":defs,"examples":ex})
    # sub-entries & proverbs appear in collapsed blocks
    sub=[clean(x) for x in re.findall(r'class="azpisarrera[^"]*"[^>]*>(.*?)</a>', h, re.S)]
    prov=[]
    i=h.find('Esaera zaharrak')
    out[w]={"entries":entries,"subentries":sub[:30]}
    n=sum(len(e["examples"]) for e in entries)
    print(f"{w:16s} entries={len(entries)} ex={n} | " + " || ".join(f'{e["head"]} [{",".join(e["marks"])}] {e["defs"][:1]} -> {e["examples"][:2]}' for e in entries)[:400])
json.dump(out, open("eh_examples.json","w"), ensure_ascii=False, indent=1)
