import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dropout, LSTM, Dense, Add, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.last_attention_weights = None

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        self.last_attention_weights = attention_weights
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_model_with_attention(vocab_size, max_length, embedding_dim=256, units=256):
    inputs_text = Input(shape=(max_length,), name="input_text")
    text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs_text)
    text_embedding = Dropout(0.5)(text_embedding)
    lstm = LSTM(units, return_sequences=True, return_state=True)
    lstm_output, state_h, state_c = lstm(text_embedding)

    inputs_image = Input(shape=(64, 2048), name="input_image")
    attention = BahdanauAttention(units)
    context_vector, attention_weights = attention(inputs_image, state_h)

    #Об'єднання
    merged = Concatenate()([context_vector, state_h])

    dense1 = Dense(units, activation='relu')(merged)
    dense1 = Dropout(0.5)(dense1)

    outputs = Dense(vocab_size, activation='softmax')(dense1)

    model = Model(inputs=[inputs_text, inputs_image], outputs=outputs)
    return model
