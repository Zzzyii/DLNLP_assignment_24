import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, MultiHeadAttention, Dropout, LayerNormalization, Dense
from tensorflow.keras import layers

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, Dense, Dropout, Input,GlobalMaxPooling1D
 

# Define a custom Transformer encoder class, inherited from tensorflow's Layer class
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        # Initialise TransformerEncoder with super.
        super(TransformerEncoder, self).__init__()
        # MultiHeadAttention Mechanisms
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # feed-forward neural network
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        # First layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        # Second layer normalization
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        # First dropout layer
        self.dropout1 = Dropout(rate)
        # Second dropout layer
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        # Use multiple attention mechanisms
        attn_output = self.att(inputs, inputs)
        # Apply the first dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Add the first layer of normalization
        out1 = self.layernorm1(inputs + attn_output)
        # Through feed-forward neural networks
        ffn_output = self.ffn(out1)
        # Apply the second dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # Returns the result of the second layer normalization
        return self.layernorm2(out1 + ffn_output)

# Define a function to build the model
def build_transformer_cnn_bilstm_model(vocab_size, embedding_dim, num_heads, ff_dim,sequence_length):
    inputs = Input(shape=(sequence_length,), dtype=tf.int32)
    x = Embedding(vocab_size, embedding_dim)(inputs)

    # Create a Transformer encoder block
    transformer_block = TransformerEncoder(embedding_dim, num_heads, ff_dim)
    # Through the Transformer encoder
    x = transformer_block(x)

    # Add a 1D convolutional layer
    x = layers.Conv1D(64, kernel_size=2, padding='same', activation='relu')(x)
    # Add a bi-directional LSTM layer
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    # Adding a dropout layer
    x = layers.Dropout(0.1)(x)
    # Adding a Fully Connected Layer
    x = layers.Dense(20, activation="relu")(x)
    # Add the dropout layer again
    x = layers.Dropout(0.1)(x)
    # Output layer, a fully connected layer, using a sigmoid activation function
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Create and return the model
    model = Model(inputs=inputs, outputs=outputs)
    return model