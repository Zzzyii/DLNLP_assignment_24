import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GlobalMaxPooling1D, Dense

# Building a LSTM model
def build_lstm_model(vocab_size, embedding_dim,sequence_length):
    inputs = Input(shape=(sequence_length,), dtype=tf.int32)
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = LSTM(16, return_sequences=True, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model