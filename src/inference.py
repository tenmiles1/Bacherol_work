import os
import json
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from model import build_model_with_attention as build_model

with open('data/processed/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
    wordtoix = tokenizer_data['wordtoix']
    ixtoword = {int(k): v for k, v in tokenizer_data['ixtoword'].items()}
vocab_size = len(wordtoix)

with open('data/processed/max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

model = build_model(vocab_size, max_length)
model.load_weights('data/processed/image_caption_model_best_val_loss.weights (4).h5')


cnn_model = InceptionV3(weights='imagenet', include_top=False)
cnn_model.trainable = False


def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((299, 299))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    features = cnn_model.predict(x)
    return np.reshape(features, (64, 2048))


def generate_caption_beam(model, image_feature, wordtoix, ixtoword, max_length, beam_size=3):
    sequences = [[[wordtoix['<start>']], 0.0]]
    max_loops = min(20, max_length)

    for _ in range(max_loops):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == wordtoix['<end>'] and len(seq) > 5:
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model([padded, np.array([image_feature])], training=False)[0]

            top_k = np.argsort(preds)[::-1][:beam_size]
            seen_bigrams = set(zip(seq[:-1], seq[1:]))

            for word in top_k:
                if len(seq) >= 2 and (seq[-1], word) in seen_bigrams:
                    continue

                new_seq = seq + [word]
                new_score = score + np.log(preds[word] + 1e-10) - 0.1 * len(seq)
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

        if sequences[0][0][-1] == wordtoix['<end>'] and len(sequences[0][0]) > 5:
            break

    final_seq = sequences[0][0]

    if wordtoix['<end>'] in final_seq:
        final_seq = final_seq[:final_seq.index(wordtoix['<end>'])]

    caption = [ixtoword.get(idx, '') for idx in final_seq if
               idx > 0 and ixtoword.get(idx) not in ['<start>', '<end>', '<unk>']]
    return ' '.join(caption)


def main():
    img_path = 'data/raw/1-fd02a124.png'
    features = extract_features(img_path)
    caption = generate_caption_beam(model, features, wordtoix, ixtoword, max_length, beam_size=3)
    print("\nPredicted Caption:", caption)


if __name__ == '__main__':
    main()
