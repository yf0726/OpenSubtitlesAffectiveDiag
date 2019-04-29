# Bahdanaua source code intro: https://zhuanlan.zhihu.com/p/43646041

import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
#from my_attention_wrapper import BahdanauAttention

def _my_bahdanau_score(processed_query, keys, attention_v):
    processed_query = tf.expand_dims(processed_query, 1) # add one demension
    # keys: h query: s attention_v: v_a
    return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query), [2]) 

def _my_affective_attention_score(query, values, attention_Wb,embedding, VAD, enc_input_tf):
    # query and values are hidden state of decoder and encoder (without multiplying W as in bahdanau_score)
    # query : [batch_size, n_hidden_units_dec]
    # values: [batch_size,max_utterance_len, n_hidden_units_enc]
    # embedding: [batch_size,max_utterance_len, embedding_size]
    # VAD: [batch_size,max_utterance_len, VAD_size]
    # enc_input_tf: [batch_size,max_utterance_len, 1]
    # WB: embedding_size -> VAD_size
    gamma = 0.5
    
    query = tf.expand_dims(query, 1)
    zeros = tf.zeros([embedding.shape[0], 1, embedding.shape[2]], dtype=tf.float32)
    concat = tf.concat([zeros, embedding], 1)
#     x_t_1 = tf.slice(concat, [0, 0, 0], [-1, embedding.shape[1], -1])
    x_t_1 = embedding
    # x_t_1: [batch_size,max_utterance_len, embedding_size]
    # beta: [batch_size,max_utterance_len, VAD_size]
    beta = tf.tanh(attention_Wb.apply(x_t_1)) # beta nan
    # yita = enc_input_tf * tf.multiply(1+beta, VAD)
    yita = enc_input_tf * tf.multiply(1+beta,VAD) # [batch_size,max_utterance_len,3]
#     yita = enc_input_tf * VAD
#     yita = gamma * tf.square(tf.norm(yita,ord = 2,axis = 2)) # [batch_size,max_utterance_len]
    yita = gamma * tf.reduce_sum(tf.square(yita),[2])
    # expand the second demension of query, and on that demension duplicate [batch_size, n_hidden_units_dec] when multiply with values
    # keys: h_t_1 query: s_t attention_v: v_a
    e_t_t = tf.reduce_sum(values * query, [2])
    return e_t_t + yita # affective attention?
    # after reduce_sum the size is [batch_size,max_utterance_len]

class MyBahdanauAttention(BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length = None,
                 enc_input_embed = None,
                 enc_input_VAD = None,
                 enc_input_tf = None
                 ):
        super(MyBahdanauAttention, self).__init__(
            num_units = num_units,
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            name = 'MyBahdanauAttention')
        self.enc_input_embed = enc_input_embed
        self.enc_input_VAD = enc_input_VAD
        self.enc_input_tf = enc_input_tf
        # Generate from a truncated normal distribution. 
        # Truncated means values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked
        # num_units = 128
        self._attention_v = tf.Variable(tf.truncated_normal([num_units], stddev = 0.08), name = 'attention_v')
        self._attention_Wb = tf.layers.Dense(units=3,
                                             use_bias=False,
                                             kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1),
                                             name='attention_Wb')

    def __call__(self, query, state):
        # 首先是使用_prepare_memory函数对memory进行处理，
        # 然后memory_layer对memory进行全连接的维度变换，变换成[batch_size, max_time, num_units]
        # [1,_,128]
        # _key函数表示Encoder的输出，也即是memory的变换后的值 h
        # procesed_query值为decoder 隐藏层 s
        # keys and query size: ?
        with tf.variable_scope(None, 'my_bahdanau_attention', [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _my_affective_attention_score(query, self._values, self._attention_Wb, self.enc_input_embed,self.enc_input_VAD,self.enc_input_tf) # affect rich
#             score = _my_bahdanau_score(processed_query, self._keys, self._attention_v) #e_ij
        alignments = self._probability_fn(score, state) # compute softmax? 
        # probability_fn：A callable function which converts the score to probabilities. 
        # 计算概率时的函数，必须是一个可调用的函数，默认使用 softmax()
        next_state = alignments # alpha
        # shape=(1, len(encoder_sentence))
        return alignments, next_state # (1,19,256)
