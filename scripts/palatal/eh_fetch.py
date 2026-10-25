import re, html, sys, json, time, subprocess, os
UA="Mozilla/5.0 (X11; Linux x86_64) Firefox/130.0"
words=sys.argv[1:]
out={}
os.makedirs("eh", exist_ok=True)
for w in words:
    f=f"eh/{w}.html"
    if not os.path.exists(f):
        url=f"https://www.euskaltzaindia.eus/index.php?option=com_hiztegianbilatu&task=bilaketa&Itemid=1693&lang=eu-ES&non=sarreraBuruaStrict&query={w}"
        subprocess.run(["curl","-s","-L","-A",UA,"-c","cj.txt","-b","cj.txt",url,"-o",f],check=False)
        time.sleep(1.0)
    h=open(f,encoding="utf-8",errors="ignore").read()
    h=re.sub(r'<script.*?</script>|<style.*?</style>','',h,flags=re.S)
    m=re.search(r'emaitza .*? bilaketarentzat.*?bistaratzen\.(.*?)Aztergaia', h, re.S)
    if "0 emaitza" in h or not m:
        out[w]={"absent":True}; print(w, "ABSENT"); continue
    seg=m.group(1)
    body=html.unescape(re.sub(r'<[^>]+>',' ',seg)); body=re.sub(r'\s+',' ',body).strip()
    ital=[html.unescape(re.sub('<[^>]+>','',e)).strip() for e in re.findall(r'<(?:em|i)[^>]*>(.*?)</(?:em|i)>', seg, re.S)]
    ital=[x for x in ital if len(x)>8]
    out[w]={"body":body[:1500],"examples":ital}
    print(w, len(ital), "ex |", body[:160])
json.dump(out, open("eh_examples.json","w"), ensure_ascii=False, indent=1)
