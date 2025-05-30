import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from model import build_model_with_attention
from data_generator import DataGenerator

def main():
    processed_dir = 'data/processed'

    with open(os.path.join(processed_dir, 'cleaned_captions.pkl'), 'rb') as f:
        cleaned_captions = pickle.load(f)
    with open(os.path.join(processed_dir, 'max_length.pkl'), 'rb') as f:
        max_length = pickle.load(f)
    with open(os.path.join(processed_dir, 'features.pkl'), 'rb') as f:
        features_train = pickle.load(f)
    with open(os.path.join(processed_dir, 'features_val.pkl'), 'rb') as f:
        features_val = pickle.load(f)

    with open("data/splits/Flickr_30k.trainImages.txt", "r") as f:
        train_ids = [line.strip().replace('.jpg', '') for line in f.readlines()]
    with open("data/splits/Flickr_30k.devImages.txt", "r") as f:
        val_ids = [line.strip().replace('.jpg', '') for line in f.readlines()]

    with open(os.path.join(processed_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        wordtoix = tokenizer_data["wordtoix"]
        ixtoword = {int(k): v for k, v in tokenizer_data["ixtoword"].items()}
    vocab_size = len(wordtoix)

    common_train_ids = [img_id for img_id in train_ids if img_id in cleaned_captions and img_id in features_train]
    common_val_ids = [img_id for img_id in val_ids if img_id in cleaned_captions and img_id in features_val]

    filtered_captions_train = {k: cleaned_captions[k] for k in common_train_ids}
    filtered_features_train = {k: features_train[k] for k in common_train_ids}
    filtered_captions_val = {k: cleaned_captions[k] for k in common_val_ids}
    filtered_features_val = {k: features_val[k] for k in common_val_ids}

    print(f"Залишено {len(filtered_captions_train)} train прикладів і {len(filtered_captions_val)} val прикладів")

    train_gen = DataGenerator(
        filtered_captions_train, filtered_features_train, wordtoix, max_length,
        batch_size=256, noise_prob=0.1
    )
    val_gen = DataGenerator(
        filtered_captions_val, filtered_features_val, wordtoix, max_length,
        batch_size=256, noise_prob=0.0
    )
    vocab_size = len(wordtoix)
    print("Vocab size:", len(wordtoix))

    model = build_model_with_attention(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=256,
        units=256
    )
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(os.path.join(processed_dir, 'image_caption_model_best_val_loss.weights.h5'),
                        save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        CSVLogger("training_log.csv", append=False)
    ]

    # print("Приклад генератора:")
    # (x_text, x_img), y = val_gen[0]
    # print("x_text shape:", x_text.shape)
    # print("x_img shape:", x_img.shape)
    # print("y shape:", y.shape)
    # print("y[:10]:", y[:10])
    # print("y contains NaN:", np.isnan(y).any())
    # print("y contains Inf:", np.isinf(y).any())
    # print("y max:", np.max(y), "— vocab size:", vocab_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=5.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=35,
        callbacks=callbacks
    )

if __name__ == '__main__':
    main()