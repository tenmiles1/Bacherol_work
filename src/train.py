import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import build_model


def main():
    processed_dir = 'data/processed'

    cleaned_captions_file = os.path.join(processed_dir, 'cleaned_captions.pkl')
    vocab_file = os.path.join(processed_dir, 'vocab.pkl')
    max_length_file = os.path.join(processed_dir, 'max_length.pkl')

    features_file = os.path.join(processed_dir, 'features.pkl')


    with open(cleaned_captions_file, 'rb') as f:
        cleaned_captions = pickle.load(f)

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    with open(max_length_file, 'rb') as f:
        max_length = pickle.load(f)

    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    print(f"Завантажено cleaned_captions (keys: {len(cleaned_captions)}), "
          f"vocab (size: {len(vocab)}), max_length={max_length}, "
          f"features (keys: {len(features)})")


    vocab_list = sorted(list(vocab))
    wordtoix = {w: i + 1 for i, w in enumerate(vocab_list)}

    wordtoix['<pad>'] = 0

    start_token = '<start>'
    end_token = '<end>'

    if start_token not in wordtoix:
        wordtoix[start_token] = len(wordtoix)
    if end_token not in wordtoix:
        wordtoix[end_token] = len(wordtoix)

    vocab_size = len(wordtoix)
    print(f"Кінцевий vocab_size (з спецтокенами): {vocab_size}")

    Xtext = []
    Ximage = []
    y = []

    def caption_to_seq(caption):
        words = caption.split()
        return [wordtoix[w] if w in wordtoix else 0 for w in words]

    for img_name, caps_list in cleaned_captions.items():
        if img_name not in features:
            continue

        img_vector = features[img_name]  # (2048,)

        for caption in caps_list:
            seq = [wordtoix[start_token]] + caption_to_seq(caption) + [wordtoix[end_token]]

            for i in range(1, len(seq)):
                partial = seq[:i]
                next_word = seq[i]

                partial_padded = pad_sequences([partial], maxlen=max_length, padding='post')[0]

                Xtext.append(partial_padded)
                Ximage.append(img_vector)
                y.append(next_word)

    Xtext = np.array(Xtext)
    Ximage = np.array(Ximage)
    y = np.array(y)

    print(f"Створено {len(Xtext)} прикладів для тренування.")
    print(f"Xtext.shape = {Xtext.shape}, Ximage.shape = {Ximage.shape}, y.shape = {y.shape}")

    model = build_model(vocab_size=vocab_size, max_length=max_length, embedding_dim=256, units=256)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.summary()

    model.fit(
        [Xtext, Ximage],
        y,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )

    model.save(os.path.join(processed_dir, 'image_caption_model.h5'))
    print(f"Model saved to {os.path.join(processed_dir, 'image_caption_model.h5')}")


if __name__ == '__main__':
    main()
