import os
import re
import random
import pickle
from collections import defaultdict


def load_captions(filename):
    captions = defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as file:
        header = next(file).strip()
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) < 2:
                print(f"Рядок з неправильним форматом: {line}")
                continue
            image_id, caption = parts[0], parts[1]
            captions[image_id].append(caption)
    return captions


def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    return caption.strip()


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
    max_length = 0
    for caps in captions_dict.values():
        for caption in caps:
            length = len(caption.split())
            if length > max_length:
                max_length = length
    return max_length


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

    with open(os.path.join(output_dir, 'Flickr_8k.trainImages.txt'), 'w', encoding='utf-8') as f:
        for img in train_ids:
            f.write(f"{img}\n")

    with open(os.path.join(output_dir, 'Flickr_8k.devImages.txt'), 'w', encoding='utf-8') as f:
        for img in val_ids:
            f.write(f"{img}\n")

    with open(os.path.join(output_dir, 'Flickr_8k.testImages.txt'), 'w', encoding='utf-8') as f:
        for img in test_ids:
            f.write(f"{img}\n")

    return train_ids, val_ids, test_ids


if __name__ == '__main__':
    captions_file = 'data/raw/Flickr8k/captions.txt'

    captions = load_captions(captions_file)

    cleaned_captions = preprocess_captions(captions)

    vocab = build_vocabulary(cleaned_captions)

    max_length = get_max_caption_length(cleaned_captions)

    print("Загальна кількість зображень:", len(cleaned_captions))
    print("Розмір словника:", len(vocab))
    print("Максимальна довжина підпису:", max_length)

    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    cleaned_captions_file = os.path.join(processed_dir, 'cleaned_captions.pkl')
    with open(cleaned_captions_file, 'wb') as f:
        pickle.dump(cleaned_captions, f)
    print("Очищені підписи збережено в файл:", cleaned_captions_file)

    vocab_file = os.path.join(processed_dir, 'vocab.pkl')
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print("Словник збережено в файл:", vocab_file)

    max_length_file = os.path.join(processed_dir, 'max_length.pkl')
    with open(max_length_file, 'wb') as f:
        pickle.dump(max_length, f)
    print("max_length збережено в файл:", max_length_file)


    all_image_ids = set(cleaned_captions.keys())
    train_ids, val_ids, test_ids = split_and_save_image_ids(all_image_ids)

    print("Кількість зображень для навчання:", len(train_ids))
    print("Кількість зображень для валідації:", len(val_ids))
    print("Кількість зображень для тестування:", len(test_ids))
