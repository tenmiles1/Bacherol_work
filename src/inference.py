import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import build_model


def load_vocab_and_tokenizer():
    processed_dir = 'data/processed'
    vocab_path = os.path.join(processed_dir, 'vocab.pkl')

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_list = sorted(list(vocab))
    wordtoix = {w: i + 1 for i, w in enumerate(vocab_list)}
    wordtoix['<pad>'] = 0

    if '<start>' not in wordtoix:
        wordtoix['<start>'] = len(wordtoix)
    if '<end>' not in wordtoix:
        wordtoix['<end>'] = len(wordtoix)

    ixtoword = {v: k for k, v in wordtoix.items()}

    vocab_size = len(wordtoix)
    return wordtoix, ixtoword, vocab_size


def load_max_length():
    processed_dir = 'data/processed'
    max_length_path = os.path.join(processed_dir, 'max_length.pkl')
    with open(max_length_path, 'rb') as f:
        max_length = pickle.load(f)
    return max_length


def load_trained_model(vocab_size, max_length):
    processed_dir = 'data/processed'
    model_path = os.path.join(processed_dir, 'image_caption_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file wasnt found: {model_path}")

    model = build_model(vocab_size=vocab_size, max_length=max_length, embedding_dim=256, units=256)
    model.load_weights(model_path)
    return model


def get_inception_model():
    inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    return inception


def extract_features(img_path, inception_model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = inception_model.predict(x)
    return features[0]  # (2048,)


def generate_caption_greedy(model, photo_features, wordtoix, ixtoword, max_length):
    start_token = wordtoix.get('<start>')
    end_token = wordtoix.get('<end>')

    in_text = [start_token]

    for _ in range(max_length):
        sequence = pad_sequences([in_text], maxlen=max_length, padding='post')
        yhat = model.predict([sequence, np.array([photo_features])], verbose=0)
        yhat = np.argmax(yhat)
        in_text.append(yhat)

        if yhat == end_token:
            break

    caption_words = []
    for idx in in_text:
        word = ixtoword.get(idx, '')
        if word == '<start>' or word == '<pad>':
            continue
        if word == '<end>':
            break
        caption_words.append(word)

    final_caption = ' '.join(caption_words)
    return final_caption


def main():
    wordtoix, ixtoword, vocab_size = load_vocab_and_tokenizer()
    max_length = load_max_length()
    model = load_trained_model(vocab_size, max_length)
    model.summary()
    inception_model = get_inception_model()

    test_image_path = 'data/raw/Flickr8k/Images/216172386_9ac5356dae.jpg'
    photo_features = extract_features(test_image_path, inception_model)
    caption = generate_caption_greedy(model, photo_features, wordtoix, ixtoword, max_length)
    print("Згенерований підпис:", caption)
    pass


if __name__ == '__main__':
    main()