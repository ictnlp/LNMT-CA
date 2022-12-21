python3 save_word_attn.py multi30k-dataset/data-bin/multilingual_share \
  --gen-subset test --task translation \
  --source-lang mul --target-lang en \
  --dest fairseq/result \
  --lang de

python3 save_word_attn.py multi30k-dataset/data-bin/multilingual_share \
  --gen-subset test1 --task translation \
  --source-lang mul --target-lang en \
  --dest fairseq/result  \
  --lang fr

python3 save_word_attn.py multi30k-dataset/data-bin/multilingual_share \
  --gen-subset test2 --task translation \
  --source-lang mul --target-lang en \
  --dest fairseq/result \
  --lang cs