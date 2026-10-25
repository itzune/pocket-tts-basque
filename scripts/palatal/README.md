# Palatal word / sentence mining

Builds `resources/palatal/palatal_words.tsv` and `resources/palatal/palatal_sentences.tsv`
(Basque words with the palatal digraphs tt, dd, ll and example sentences for them).

Run from a scratch directory, with the repo root as the output base:

1. `wiki_scan.py` — scans the eu.wikipedia parquet dump (`wikimedia/wikipedia`, `20231101.eu`,
   downloaded with `huggingface_hub`) and writes `wiki_pal.json` (word frequencies + sample sentences).
2. `eh_fetch.py WORD...` — fetches Euskaltzaindiaren Hiztegia entries (one request per second) into `eh/`;
   `eh_parse.py` extracts headword, register marks, definitions and the italic example sentences into `eh_examples.json`.
   The full headword lists came from the dictionary's "du" (contains) search for `dd`, `tt`, `ll`.
3. EEH (EHU) and Elhuyar examples were collected by hand into `eeh_elhuyar_raw.txt` (`=== word ===` / source line / `- sentence`).
4. `assemble.py` writes the curated word list; `assemble_sents.py` merges all sources into the sentence bank.
