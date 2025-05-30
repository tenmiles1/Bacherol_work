import pickle
import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer

with open('data/processed/cleaned_captions.pkl', 'rb') as f:
    cleaned_captions = pickle.load(f)

all_captions = []
for caps in cleaned_captions.values():
    all_captions.extend(caps)

tokenizer = Tokenizer(oov_token='<unk>', filters='', lower=True)
tokenizer.fit_on_texts(all_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

idx = max(tokenizer.word_index.values()) + 1
if '<start>' not in tokenizer.word_index:
    tokenizer.word_index['<start>'] = idx
    tokenizer.index_word[idx] = '<start>'
    idx += 1
if '<end>' not in tokenizer.word_index:
    tokenizer.word_index['<end>'] = idx
    tokenizer.index_word[idx] = '<end>'

wordtoix = tokenizer.word_index
ixtoword = tokenizer.index_word

tokenizer_dict = {
    "wordtoix": wordtoix,
    "ixtoword": ixtoword
}

output_path = 'data/processed/tokenizer.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_dict, f, indent=2, ensure_ascii=False)

print(f"Tokenizer saved to {output_path}")