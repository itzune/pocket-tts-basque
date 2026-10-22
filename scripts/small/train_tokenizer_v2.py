import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="/root/data/bssd_v2/tokenizer/corpus_v2.txt", model_prefix="/root/data/bssd_v2/tokenizer/basque_v2",
    vocab_size=4000, character_coverage=1.0, model_type="unigram",
    normalization_rule_name="identity", remove_extra_whitespaces=False, max_sentencepiece_length=6,
    allow_whitespace_only_pieces=True, split_digits=True, byte_fallback=True, pad_id=3,
    input_sentence_size=10_000_000, shuffle_input_sentence=True,
    user_defined_symbols=["tt", "dd", "ll"],  # Basque palatal digraphs: one piece each, wherever they occur
)
sp = spm.SentencePieceProcessor(model_file="/root/data/bssd_v2/tokenizer/basque_v2.model")
print("pieces", sp.get_piece_size())
for s in ("Antton eta Maddi onddo bila joan dira basora.", "Pello Iruñean bizi da; andereñoa ere bai.", "Bihar goizeko hamarretan hasiko da bilera."):
    print(sp.encode(s, out_type=str))
