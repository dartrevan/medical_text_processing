import tensorflow_hub as hub
from keras import layers
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer


class selfAttention(Layer):
    def __init__(self, hidden_dim, **kwargs):
        self.hidden_dim = hidden_dim
        super(selfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_omega = self.add_weight(name='w_omega', shape=(input_shape[2], self.hidden_dim), initializer='uniform',
                                       trainable=True)
        self.b_omega = self.add_weight(name='b_omega', shape=(self.hidden_dim,), initializer='uniform',
                                       trainable=True)
        self.u_omega = self.add_weight(name='u_omega', shape=(self.hidden_dim,), initializer='uniform',
                                       trainable=True)
        super(selfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)


def ElmoEmbedding(x):
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)),
                      signature="default",
                      as_dict=True)["elmo"]


def build_model(num_labels, max_length, embedding_matrix, elmo_embeddings=False):
    input_text = layers.Input(shape=(max_length,), dtype=tf.int32)

    if elmo_embeddings:
        embedding = layers.Lambda(ElmoEmbedding, output_shape=(max_length, 1024))(input_text)
    else:
        embedding = layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                                     input_length=max_length, trainable=False)(input_text)

    x = layers.Bidirectional(layers.GRU(units=128, return_sequences=True,
                                        recurrent_dropout=0.3, dropout=0.3))(embedding)

    x = selfAttention(16)(x)
    out = layers.Dense(num_labels, activation="softmax")(x)
    model = Model(input_text, out)
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

