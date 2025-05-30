import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from model import build_model_with_attention as build_model

with open('data/processed/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
    wordtoix = tokenizer_data['wordtoix']
    ixtoword = {int(k): v for k, v in tokenizer_data['ixtoword'].items()}


with open('data/processed/max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)


vocab_size = len(wordtoix)
model = build_model(vocab_size, max_length)
model.load_weights('data/processed/image_caption_model (1).h5')

cnn_model = InceptionV3(weights='imagenet', include_top=False)

def extract_features(img):
    print("(DEBUG) Extracting visual features from image")
    img = img.resize((299, 299))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = cnn_model.predict(x)
    features = np.reshape(features, (64, 2048))
    print("(DEBUG) Feature shape:", features.shape)
    return features

def generate_caption_beam(model, image_feature, wordtoix, ixtoword, max_length, beam_size=3):
    print("(DEBUG) Starting beam search generation")
    sequences = [[[wordtoix['<start>']], 0.0]]
    loop_count = 0
    max_loops = 50

    while True:
        loop_count += 1
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == wordtoix['<end>'] or len(seq) >= max_length:
                all_candidates.append((seq, score))
                continue
            padded = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model([padded, np.array([image_feature])], training=False)[0]
            top_k = np.argsort(preds)[-beam_size:]
            for word in top_k:
                new_seq = seq + [word]
                new_score = score + np.log(preds[word] + 1e-10)
                all_candidates.append((new_seq, new_score))
        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

        print(f"(DEBUG) Loop {loop_count}, best sequence so far: {[ixtoword.get(i, '') for i in sequences[0][0]]}")

        if sequences[0][0][-1] == wordtoix['<end>']:
            break
        if loop_count >= max_loops:
            print("Exceeded max beam steps. breaking")
            break

    final_seq = sequences[0][0]
    result = [ixtoword.get(idx, '') for idx in final_seq if idx > 0 and idx != wordtoix['<end>']]
    print("(DEBUG) Final caption:", result)
    return ' '.join(result)

st.title("Image Captioning")
st.write("Завантажте зображення і модель згенерує підпис")

uploaded_file = st.file_uploader("Оберіть зображення", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Завантажене зображення", use_container_width=True)

    if st.button("Згенерувати підпис"):
        with st.spinner('Генерація..'):
            features = extract_features(image)
            caption = generate_caption_beam(model, features, wordtoix, ixtoword, max_length, beam_size=3)
        st.success("Готово!")
        st.markdown(f"Caption: {caption}")