# Bahdanaua source code intro: https://zhuanlan.zhihu.com/p/43646041

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention

def _my_bahdanau_score(processed_query, keys, attention_v):
    processed_query = tf.expand_dims(processed_query, 1) # add one demension
    # keys: h query: s
    return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query), [2]) # affective attention?

class MyBahdanauAttention(BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length = None):
        super(MyBahdanauAttention, self).__init__(
            num_units = num_units,
            memory = memory,# memory = enc_outputs
            memory_sequence_length = memory_sequence_length,
            name = 'MyBahdanauAttention')
        # Generate from a truncated normal distribution. 
        # Truncated means values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked
        self._attention_v = tf.Variable(tf.truncated_normal([num_units], stddev = 0.1), name = 'attention_v')

    def __call__(self, query, state):
        with tf.variable_scope(None, 'my_bahdanau_attention', [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            # what is keys? and processdd query(s_t-1)?
            score = _my_bahdanau_score(processed_query, self._keys, self._attention_v) #e_ij
        alignments = self._probability_fn(score, state) # compute softmax? 
        next_state = alignments # alpha
        return alignments, next_state
