"""

"""

import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class PointWiseFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, rate=0.1):
        super(PointWiseFFN, self).__init__()

        self.conv1d_1 = tf.keras.layers.Conv1D(d_model, kernel_size=1, activation='relu')
        self.conv1d_2 = tf.keras.layers.Conv1D(d_model, kernel_size=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        out = self.conv1d_1(x)
        out = self.dropout(out, training=training)
        out = self.conv1d_2(out)

        return out


# def point_wise_feed_forward_network(d_model, training, dropout_rate=0.2):
#     # self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
#     # self.dropout1 = torch.nn.Dropout(p=dropout_rate)
#     # self.relu = torch.nn.ReLU()
#     # self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
#     # self.dropout2 = torch.nn.Dropout(p=dropout_rate)
#
#     return tf.keras.Sequential([
#         tf.keras.layers.Conv1D(d_model, kernel_size=1, activation='relu'),
#         tf.keras.layers.Dropout(rate=dropout_rate, trainable=training),# (batch_size, seq_len, dff)
#         tf.keras.layers.Conv1D(d_model, kernel_size=1, activation='relu') # input_shape=input_shape[1:]
#         # tf.keras.layers.Dropout(rate=dropout_rate)  # (batch_size, seq_len, dff)
#     ])


class SASEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(SASEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFFN(d_model, rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1, training=training)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class SASEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(SASEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [SASEncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


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
    def __init__(self, is_input, num_intents, num_layers, max_len, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(DisentangledSeqEncoder, self).__init__()

        self.d_model = d_model

        self.is_input = is_input

        self.sas_encoder = SASEncoder(num_layers, d_model, num_heads, dff, input_vocab_size,
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
                                        tf.keras.initializers.RandomNormal(mean=0., stddev=1./tf.math.sqrt(
                                            tf.cast(d_model, tf.float32))))
                               for _ in range(num_intents)
                               ]
        self.beta_input_seq = tf.concat(self.beta_input_seq, 1)
        self.beta_input_seq = tf.transpose(self.beta_input_seq)

        self.beta_label_seq = [BiasLayer(d_model,
                                        tf.keras.initializers.RandomNormal(mean=0., stddev=1./tf.math.sqrt(
                                            tf.cast(d_model, tf.float32))))
                               for _ in range(num_intents)
                               ]
        self.beta_label_seq = tf.concat(self.beta_label_seq, 1)
        self.beta_label_seq = tf.transpose(self.beta_label_seq)

    def call(self, x, training, mask):

        z = self.sas_encoder(x, training, mask)

        # function to perform
        attention_weights_p_k_i = self.intention_clustering(x)

        attention_weights_p_i = self.intention_weighting(x)

        encoded = self.intention_aggr(attention_weights_p_k_i, attention_weights_p_i, z)

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
        prototypes_t = tf.tile(tf.expand_dims(prototypes,0), [tf.shape(z)[0], 1, 1])
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
