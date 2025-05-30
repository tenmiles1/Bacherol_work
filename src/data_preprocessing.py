import os
import re
import csv
import pickle
import random
from collections import defaultdict
import pandas as pd


def clean_caption(caption):
    if not isinstance(caption, str):
        return ""
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    return caption.strip()


def load_captions_from_csv(csv_file):
    df = pd.read_csv(csv_file, sep='|', engine='python')
    df.columns = ['image_name', 'comment_number', 'comment']
    df['image_name'] = df['image_name'].str.strip()
    df['comment'] = df['comment'].str.strip()

    captions = defaultdict(list)
    for _, row in df.iterrows():
        image_id = row['image_name']
        caption = row['comment']
        captions[image_id].append(caption)

    return captions


def preprocess_captions(captions_dict):
    cleaned_captions = {}
    for image_id, caps in captions_dict.items():
        cleaned_list = [clean_caption(c) for c in caps]
        cleaned_captions[image_id] = cleaned_list
    return cleaned_captions


def build_vocabulary(captions_dict):
    vocab = set()
    for caps in captions_dict.values():
        for caption in caps:
            words = caption.split()
            vocab.update(words)
    return vocab


def get_max_caption_length(captions_dict):
    return max(len(caption.split()) for caps in captions_dict.values() for caption in caps)


def split_and_save_image_ids(image_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir='data/splits'):
    os.makedirs(output_dir, exist_ok=True)
    image_ids = list(image_ids)
    random.shuffle(image_ids)
    total = len(image_ids)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:]

    with open(os.path.join(output_dir, 'Flickr_30k.trainImages.txt'), 'w', encoding='utf-8') as f:
        for img in train_ids:
            f.write(f"{img}\n")

    with open(os.path.join(output_dir, 'Flickr_30k.devImages.txt'), 'w', encoding='utf-8') as f:
        for img in val_ids:
            f.write(f"{img}\n")

    with open(os.path.join(output_dir, 'Flickr_30k.testImages.txt'), 'w', encoding='utf-8') as f:
        for img in test_ids:
            f.write(f"{img}\n")

    return train_ids, val_ids, test_ids

def preview_caption_cleaning(captions_dict, n=5):
    print("Попередній перегляд очищення підписів:")
    for i, (img, caps) in enumerate(captions_dict.items()):
        print(f"⮞ {img}")
        for cap in caps:
            cleaned = clean_caption(cap)
            print(f"   Оригінал : {cap}")
            print(f"   Очищено  : {cleaned}")
        print("-" * 40)
        if i >= n - 1:
            break


if __name__ == '__main__':
    csv_file = 'data/raw/results.csv'

    captions = load_captions_from_csv(csv_file)
    preview_caption_cleaning(captions)

    cleaned_captions = preprocess_captions(captions)
    vocab = build_vocabulary(cleaned_captions)
    max_length = get_max_caption_length(cleaned_captions)

    print("Загальна кількість зображень:", len(cleaned_captions))
    print("Розмір словника:", len(vocab))
    print("Максимальна довжина підпису:", max_length)

    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    with open(os.path.join(processed_dir, 'cleaned_captions.pkl'), 'wb') as f:
        pickle.dump(cleaned_captions, f)
    with open(os.path.join(processed_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    with open(os.path.join(processed_dir, 'max_length.pkl'), 'wb') as f:
        pickle.dump(max_length, f)

    all_image_ids = set(cleaned_captions.keys())
    train_ids, val_ids, test_ids = split_and_save_image_ids(all_image_ids)

    print("Кількість зображень для навчання:", len(train_ids))
    print("Кількість зображень для валідації:", len(val_ids))
    print("Кількість зображень для тестування:", len(test_ids))
    print(f"(DEBUG) Кількість raw зображень: {len(captions)}")
    print(f"(DEBUG) Кількість очищених зображень: {len(cleaned_captions)}")
    print(f"(DEBUG) Розмір словника: {len(vocab)}")
    print(f"(DEBUG) Максимальна довжина підпису: {max_length}")