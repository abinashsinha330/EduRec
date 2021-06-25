"""
"""
import tensorflow as tf

from sasrec_model import SASEncoder


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, initializer):
        super(BiasLayer, self).__init__()

        self.bias = self.add_weight('bias',
                                    shape=d_model,
                                    initializer=initializer,
                                    trainable=True)

    def call(self, x):
        return tf.nn.bias_add(x, self.bias)


class DisentangledSeqEncoder(tf.keras.layers.Layer):
    def __init__(self, num_intents, num_layers, max_len, d_model, num_heads, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(DisentangledSeqEncoder, self).__init__()

        self.d_model = d_model

        self.sas_encoder = SASEncoder(num_layers, d_model, num_heads, input_vocab_size,
                                      maximum_position_encoding, rate)

        # prototypical intention vector for each intention
        self.prototypes = [BiasLayer(d_model, 'zeros') for _ in range(num_intents)]
        self.prototypes = tf.concat(self.prototypes, 1)
        self.prototypes = tf.transpose(self.prototypes)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.w = tf.keras.layers.Dense(d_model)

        # self.b = BiasLayer(d_model, 'zeros')
        self.b_ = BiasLayer(d_model, 'zeros')

        # individual alpha for each position
        self.alphas = [BiasLayer(d_model, 'zeros') for _ in range(max_len)]
        self.alphas = tf.concat(self.alphas, 1)
        self.alphas = tf.transpose(self.alphas)

        self.beta_input_seq = [BiasLayer(d_model,
                                         tf.keras.initializers.RandomNormal(mean=0., stddev=1. / tf.math.sqrt(
                                             tf.cast(d_model, tf.float32))))
                               for _ in range(num_intents)
                               ]
        self.beta_input_seq = tf.concat(self.beta_input_seq, 1)
        self.beta_input_seq = tf.transpose(self.beta_input_seq)

        self.beta_label_seq = [BiasLayer(d_model,
                                         tf.keras.initializers.RandomNormal(mean=0., stddev=1. / tf.math.sqrt(
                                             tf.cast(d_model, tf.float32))))
                               for _ in range(num_intents)
                               ]
        self.beta_label_seq = tf.concat(self.beta_label_seq, 1)
        self.beta_label_seq = tf.transpose(self.beta_label_seq)

    def call(self, x, training, mask, is_input_seq):

        z = self.sas_encoder(x, training, mask)

        # function to perform
        attention_weights_p_k_i = self.intention_clustering(x)

        attention_weights_p_i = self.intention_weighting(x)

        encoded = self.intention_aggr(attention_weights_p_k_i, attention_weights_p_i, z, is_input_seq)

        return encoded  # (batch_size, input_seq_len, d_model)

    def intention_clustering(self, z):
        """
        Method to measure how likely the primary intention at position i
        is related with kth latent category
        :param z:
        :param prototypes:
        :return:
        """
        z = self.layernorm1(z)
        prototypes = self.layernorm2(self.prototypes)
        prototypes_t = tf.tile(tf.expand_dims(prototypes, 0), [tf.shape(z)[0], 1, 1])
        cosine_similarity_mat = tf.matmul(z, prototypes_t, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_model, tf.float32))

        exp = tf.math.exp(cosine_similarity_mat)

        sum_across_k = tf.reduce_sum(exp, axis=1)
        all_p_k_i = exp / tf.reshape(sum_across_k, (-1, -1, 1))

        return all_p_k_i

    def intention_weighting(self, z):
        """
        Method to measure how likely primary intention at position i
        is important for predicting user's future intentions
        :param z:
        :return:
        """
        alphas = tf.tile(tf.expand_dims(self.alphas, 0), [tf.shape(z)[0], 1, 1])
        out = alphas + z
        k_ = self.layernorm3(out)
        k = k_ + self.w(k_)
        alpha_t = tf.gather(alphas, indices=[-1], axis=1)
        input_t = tf.gather(z, indices=[-1], axis=1)
        b_ = tf.tile(tf.expand_dims(self.b_, 0), [tf.shape(z)[0], 1, 1])
        q = self.layernorm4(alpha_t + input_t + self.b_)

        cosine_similarity_mat = tf.matmul(k, q, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_model, tf.float32))

        exp = tf.math.exp(cosine_similarity_mat)

        sum_across_t = tf.reduce_sum(exp, axis=1)
        all_p_i = exp / tf.reshape(sum_across_t, (-1, -1, 1))

        return all_p_i

    def intention_aggr(self, attention_weights_p_k_i, attention_weights_p_i, z):
        """
        Method to aggregate intentions collected at all positions according
        to both kinds of attention weights
        :param attention_weights_p_k_i:
        :param attention_weights_p_i:
        :param z:
        :return:
        """
        num_intents = tf.shape(attention_weights_p_k_i)[2]
        all_p_i = tf.tile(tf.expand_dims(attention_weights_p_i, 0), [1, 1, num_intents])
        attention_weights = tf.math.multiply(attention_weights_p_k_i, all_p_i)

        out = tf.matmul(attention_weights, z, transpose_a=True)
        if self.is_input:
            encoded = self.layernorm5(self.beta_input_seq + out)
        else:
            encoded = self.layernorm5(self.beta_label_seq + out)

        return encoded
