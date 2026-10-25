import json, re, csv, collections, io
eh=json.load(open("eh_examples.json"))
wiki=json.load(open("wiki_pal.json")); wfreq=dict(wiki["freq"])
def wf(w):
    w=w.lower(); return sum(c for f,c in wfreq.items() if f==w or f.startswith(w) and len(f)<=len(w)+6)
# ---------- curated lists ----------
# (word, digraph, category, note)
NAT_DD=[("onddo","dd","native","EH Heg.; fungus/mushroom; also onddo beltz, onddo zuri, onddo-saltsa, onddoketari"),
 ("onddo beltz","dd","native","EH subentry"),("onddo zuri","dd","native","EH subentry"),("onddo-saltsa","dd","native","EH usage note"),
 ("onddoketari","dd","native","EH"),("anddere","dd","native","EH (Ipar., jas.) lady"),("anddereder","dd","native","EH weasel"),
 ("sorddes","dd","native","EH deaf-ish"),("sorddeski","dd","native","EH"),("sorddestu","dd","native","EH"),
 ("Maddi","dd","name","Basque given name (EODA); frequent in wiki"),("Maddalen","dd","name","Basque given name (EODA)"),
 ("ddanddan","dd","dialectal","OEH: ddinddin-ddanddan (R) 'campanilla'; child speech"),("ddandda","dd","dialectal","OEH (R) child speech 'church'"),
 ("ddinddi","dd","dialectal","OEH (R-uzt) child speech 'money'"),("ddinddil","dd","dialectal","OEH v. dindil"),("ddunddu","dd","dialectal","OEH v. dundu"),
 ("ddalanddal","dd","dialectal","OEH v. dalan-dalan")]
NAT_TT=["ttipi","ttipi-ttapa","ttipitu","ttipian","ttipitik","eri ttipi","ahatettipi","ttiki-ttaka","ttantta","ttanttaka","ttotta","ttak","ttaka-ttaka","ttakun","ttapa-ttapa","ttarro","ttattar","ttau-ttau","ttentte","ttintta","ttintta bat","ttirriki-ttarraka","ttirritta","ttuku-ttuku","ttunttun","ttuntturro",
 "pottoka","pottoko","pottorro","pott","pott egin","pott eman","kuttun","kutturlio","muttiko","muttur","pittin","pittin bat","pittinka","pitta","pitta bat","pattal","pattalaldi","pattaldu","pattar","pettar","puttiko","puntta","punttu","panttupa","mottel","gutti","batto","beretter","entterka","aitzinttoan",
 "deabrutto","eguntto","gizontto","erregetto","erregepittitta","kottear"]
NAT_TT_EXTRA=[("ttonttor","tt","native","hunspell/wiki; peak, hump"),("tanttaka","tt","native","wiki; drop by drop (also theatre company)"),("ttun-ttun","tt","native","wiki; string drum / dance"),("ttarttalo","tt","name","wiki; Tartalo cyclops / publisher"),("pintto","tt","name","wiki; dog name, children's song"),("punttudun","tt","native","wiki; dotted (music)"),
 ("pottolo","tt","native","hunspell; chubby"),("herritto","tt","native","hunspell; diminutive -tto"),("menditto","tt","native","hunspell"),("liburutto","tt","native","hunspell"),("mehetto","tt","native","hunspell"),("xumetto","tt","native","hunspell"),("urdetto","tt","native","hunspell"),
 ("ttottel","tt","native","hunspell? stutterer (totel)"),("kotta","tt","native","hunspell"),("mottodun","tt","native","hunspell"),("atturri","tt","name","wiki; Adour river"),
 ("Antton","tt","name","Basque given name; very frequent"),("Anttoni","tt","name","Basque given name"),("Mattin","tt","name","Basque given name"),("Kattalin","tt","name","Basque given name"),("Battitta","tt","name","Basque given name"),("Bittor","tt","name","Basque given name"),("Pettiri","tt","name","Basque given name"),("Ttipi-Ttapa","tt","name","Basque magazine")]
NAT_LL=["pollit","polliki","llabur","llollo","llollobelar","pello","pellokeria","kokollo"]
NAT_LL_EXTRA=[("llaburbide","ll","native","hunspell"),("polliki-polliki","ll","native","reduplication"),("ollo","ll","dialectal","Bizk. oilo (hen); also place name Ollo"),("bollo","ll","dialectal","hunspell? bun"),
 ("Pello","ll","name","Basque given name; very frequent"),("Ellande","ll","name","Basque given name"),("Tafalla","ll","name","Navarre town"),("Mallabia","ll","name","Biscay town"),("Ollo","ll","name","Navarre valley"),("Olloki","ll","name","Navarre village"),("Ollaran","ll","name","Navarre"),("Allo","ll","name","Navarre town"),("Allotz","ll","name","Navarre"),("Atallu","ll","name","Navarre"),("Andollu","ll","name","Alava"),("Billabona","ll","name","Gipuzkoa town"),("Arzalluz","ll","name","surname"),("Azpillaga","ll","name","surname"),("Amillaga","ll","name","surname"),("Telleria","ll","name","surname"),("Gallastegi","ll","name","surname"),("Txillardegi","ll","name","writer"),("Txillida","ll","name","Basque spelling of Chillida"),("Ollarandar","ll","name","demonym")]
LOAN_LL=["amilla","arpillera","arrallant","arranparrilla","artillari","artilleria","bainilla","balazta-pastilla","banderilla","banilla","billar","billete","billano","bollera","bonbilla","erroskilla","eskotilla","eskuadrilla","estribillo","flotilla","folletoi","galleta","galtzontzillo","garrotillo","gerrilla","gerrillari","gillotina","gillotinatu","gresilla","karilloi","kartilla","kasulla","kotilloi","kuadrilla","lentilla","makillaje","makillatu","makillatzaile","mantzanilla","menbrillo","natilla","nobillada","paella","parrilla","parrillada","parrillero","pasillo","pastilla","patilla","pella","pikillo","plantilla","puntilla","silla","tailla","taillatu","tallarin","tempranillo","toalla","tokilla","tortilla","trainerilla","txintxilla","brillante"]
LOAN_NAMES_LL=[("Sevilla","ll","loan-name","Spanish place; pronounced palatal in Basque"),("Castilla","ll","loan-name",""),("Valladolid","ll","loan-name",""),("Mallorca","ll","loan-name",""),("Murillo","ll","loan-name",""),("Trujillo","ll","loan-name",""),("Melilla","ll","loan-name",""),("Villanueva","ll","loan-name",""),("Villa","ll","loan-name","Spanish 'villa-' names")]
def entry(w):
    e=eh.get(w.lower().replace(" ","%20")) or eh.get(w.lower()) or eh.get(w)
    if not e or not e.get("entries"): return None
    return e["entries"][0]
rows=[]
def add(w,dg,cat,note,src):
    e=entry(w); marks=",".join(e["marks"]) if e else ""; d=(e["defs"][0] if e and e["defs"] else "")
    nex=len(e["examples"]) if e else 0
    rows.append(dict(word=w,digraph=dg,category=cat,eh_marks=marks,eh_def=d[:80],eh_examples=nex,wiki_sent=wf(w.split()[0]),source=src,note=note))
for w,dg,cat,note in NAT_DD: add(w,dg,cat,note,"EH/OEH/EODA")
for w in NAT_TT: add(w,"tt","native","","EH")
for w,dg,cat,note in NAT_TT_EXTRA: add(w,dg,cat,note,"hunspell/wiki/EODA")
for w in NAT_LL: add(w,"ll","native","","EH")
for w,dg,cat,note in NAT_LL_EXTRA: add(w,dg,cat,note,"hunspell/wiki/EODA")
for w in LOAN_LL: add(w,"ll","loan-es","Spanish loan, palatal ll in Basque","EH")
for w,dg,cat,note in LOAN_NAMES_LL: add(w,dg,cat,note,"wiki")
seen=set(); rows=[r for r in rows if not (r["word"].lower() in seen or seen.add(r["word"].lower()))]
out="resources/palatal/palatal_words.tsv"
import os; os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w",newline="") as f:
    f.write("# Basque words with palatal digraphs (tt, dd, ll). Sources: Euskaltzaindiaren Hiztegia (EH, 'contains' search for dd/tt/ll),\n# Orotariko Euskal Hiztegia (OEH), EODA names, hunspell-eu, eu.wikipedia (sentence counts). category: native | dialectal | name | loan-es | loan-name.\n")
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),delimiter="\t"); w.writeheader(); w.writerows(rows)
c=collections.Counter((r["digraph"],r["category"]) for r in rows)
print(len(rows),"words"); print(sorted(c.items()))
print("missing EH entry:", [r["word"] for r in rows if r["source"]=="EH" and not r["eh_def"]])
