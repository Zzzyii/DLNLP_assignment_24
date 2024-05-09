import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Build a CNN model
def build_cnn_model(vocab_size, embedding_dim,sequence_length):
    inputs = Input(shape=(sequence_length,), dtype=tf.int32)
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = Conv1D(64, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(20, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model