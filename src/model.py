import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dropout, LSTM, Dense, Add
from tensorflow.keras.models import Model


def build_model(vocab_size, max_length, embedding_dim=256, units=256):

    inputs_text = Input(shape=(max_length,), name="input_text")

    x1 = Embedding(input_dim=vocab_size,
                   output_dim=embedding_dim,
                   mask_zero=True)(inputs_text)

    x1 = Dropout(0.5)(x1)

    x1 = LSTM(units)(x1)

    inputs_image = Input(shape=(2048,), name="input_image")
    x2 = Dropout(0.5)(inputs_image)
    x2 = Dense(units, activation='relu')(x2)

    merged = Add()([x1, x2])
    x3 = Dense(units, activation='relu')(merged)
    outputs = Dense(vocab_size, activation='softmax')(x3)

    model = Model(inputs=[inputs_text, inputs_image], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model