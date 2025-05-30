import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def add_caption_noise(tokens, unk_token='<unk>', prob=0.1):
    return [word if np.random.rand() > prob else unk_token for word in tokens]

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, captions, features, wordtoix, max_length, batch_size=64, shuffle=True, noise_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.captions = captions
        self.features = features
        self.wordtoix = wordtoix
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_prob = noise_prob
        self.image_ids = list(captions.keys())
        self.data = self._prepare_data()
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]

        x_text, x_img, y = [], [], []

        for seq, image_feat, target_word in batch_data:
            x_text.append(seq)
            x_img.append(np.copy(image_feat))
            y.append(target_word)

        y_array = np.array(y, dtype=np.int32)

        if np.any(np.isnan(y_array)) or np.any(np.isinf(y_array)):
            raise ValueError("y містить NaN або Inf")
        if np.max(y_array) >= len(self.wordtoix):
            raise ValueError(f"y містить токен > vocab_size: max={np.max(y_array)}, vocab_size={len(self.wordtoix)}")

        return (np.array(x_text), np.array(x_img)), y_array

    def _prepare_data(self):
        data = []
        for image_id, caps in self.captions.items():
            for cap in caps:
                tokens = cap.split()
                if self.noise_prob > 0:
                    tokens = add_caption_noise(tokens, prob=self.noise_prob)

                seq = [self.wordtoix['<start>']] + [
                    self.wordtoix.get(w, self.wordtoix.get('<unk>', 0)) for w in tokens
                ]
                for i in range(1, len(seq)):
                    input_seq = pad_sequences([seq[:i]], maxlen=self.max_length, padding='post')[0]
                    target_word = seq[i]
                    data.append((input_seq, self.features[image_id], target_word))
        return data

    def get_ids_for_batch(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [img_id for _, img_id, _ in batch]
