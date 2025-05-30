import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import csv


with open('data/processed/cleaned_captions.pkl', 'rb') as f:
    captions = pickle.load(f)
with open('data/processed/features_test.pkl', 'rb') as f:
    features = pickle.load(f)
with open('data/processed/max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)
with open('data/processed/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
    wordtoix = tokenizer_data['wordtoix']
    ixtoword = {int(k): v for k, v in tokenizer_data['ixtoword'].items()}
vocab_size = len(wordtoix)


from model import build_model_with_attention

model = build_model_with_attention(vocab_size, max_length)
model.load_weights('data/processed/image_caption_model_best_val_loss.weights (2).h5')

with open('data/splits/Flickr_30k.testImages.txt', 'r') as f:
    test_ids = [line.strip().replace('.jpg', '') for line in f.readlines()]
    test_ids = [img_id for img_id in test_ids if img_id in captions and img_id in features]
    test_ids = test_ids[:400]


def generate_caption_beam(model, image_feature, wordtoix, ixtoword, max_length, beam_size=3):
    sequences = [[[wordtoix['<start>']], 0.0]]
    while True:
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
        if all(seq[-1] == wordtoix['<end>'] for seq, _ in sequences):
            break
    final_seq = sequences[0][0]
    return [ixtoword.get(idx, '') for idx in final_seq if idx > 0 and ixtoword.get(idx) not in ['<start>', '<end>']]


actual, predicted, rows = [], [], []
for idx, img_id in enumerate(test_ids):
    if idx % 10 == 0:
        print(f"[{idx}/{len(test_ids)}] Обробляється зображення: {img_id}")
    image_feature = features[img_id]
    caption = generate_caption_beam(model, image_feature, wordtoix, ixtoword, max_length)
    if not caption:
        print(f"Пропущено: {img_id} (порожній підпис)")
        continue
    references = [cap.split() for cap in captions[img_id]]
    actual.append(references)
    predicted.append(caption)
    rows.append({
        'image_id': img_id,
        'generated': ' '.join(caption),
        'gt_1': captions[img_id][0],
        'gt_2': captions[img_id][1] if len(captions[img_id]) > 1 else '',
        'gt_3': captions[img_id][2] if len(captions[img_id]) > 2 else '',
        'gt_4': captions[img_id][3] if len(captions[img_id]) > 3 else '',
        'gt_5': captions[img_id][4] if len(captions[img_id]) > 4 else '',
    })


if not actual or not predicted:
    print("Немає валідних прикладів для оцінювання.")
else:
    smoothie = SmoothingFunction().method4
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    meteor_total = 0
    rouge_total = 0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for i, (ref, pred) in enumerate(zip(actual, predicted)):
        ref_texts = [' '.join(r) for r in ref]
        pred_text = ' '.join(pred)

        try:
            meteor_total += meteor_score(ref_texts, pred_text.split())
            rouge_total += scorer.score(ref_texts[0], pred_text)['rougeL'].fmeasure
        except Exception as e:
            print(f"Проблема з прикладом {i}: {e}")
            continue

        if i % 100 == 0:
            print(f"Метрики: оброблено {i} прикладів...")

    meteor_avg = meteor_total / len(actual)
    rouge_avg = rouge_total / len(actual)

    with open("results_beam.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'generated', 'gt_1', 'gt_2', 'gt_3', 'gt_4', 'gt_5'])
        writer.writeheader()
        writer.writerows(rows)

    print("\nEvaluation on Test Set (BEAM SEARCH)")
    print(f"Test Images: {len(test_ids)}")
    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor_avg:.4f}, ROUGE-L: {rouge_avg:.4f}")
    print("Результати збережено в results_beam.csv")

