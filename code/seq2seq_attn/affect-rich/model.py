import os
import numpy as np
import tensorflow as tf
from attention import MyBahdanauAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder, dynamic_decode, sequence_loss, tile_batch

class Options(object):
    '''Parameters used by the Seq2SeqAttn model.'''
    def __init__(self, mode, num_epochs, batch_size, learning_rate, beam_width,
                 vocab_size, max_uttr_len_enc, max_uttr_len_dec, go_index, eos_index,
                 word_embed_size, n_hidden_units_enc, n_hidden_units_dec, attn_depth,
                 word_embeddings,corpus_size):
        super(Options, self).__init__()

        self.mode = mode
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beam_width = beam_width

        self.vocab_size = vocab_size # vocabulary of whole corpus?
        self.max_uttr_len_enc = max_uttr_len_enc
        self.max_uttr_len_dec = max_uttr_len_dec
        self.go_index = go_index
        self.eos_index = eos_index

        self.word_embed_size = word_embed_size
        self.n_hidden_units_enc = n_hidden_units_enc # h 
        self.n_hidden_units_dec = n_hidden_units_dec
        self.attn_depth = attn_depth # v in attention

        self.word_embeddings = word_embeddings
        self.corpus_size = corpus_size

def compute_attention(attention_mechanism, cell_output):
    alignments, _ = attention_mechanism(cell_output, None) # 每一个encoder的词对应的attention?
    expanded_alignments = tf.expand_dims(alignments, 1) # (1,19) -> (1,1,19)
    context = tf.matmul(expanded_alignments, attention_mechanism.values) # (1,1,19),(1,19,256) -> (1,1,256)
    context = tf.squeeze(context, [1])
    return context # shape (1,256)

class Seq2SeqAttn(object):
    '''Sequence to sequence network with attention mechanism.'''
    def __init__(self, options):
        super(Seq2SeqAttn, self).__init__()
        self.options = options
        self.build_graph()
        self.session = tf.Session(graph = self.graph)

    def __del__(self):
        self.session.close()
        print('TensorFlow session is closed.')

    def build_graph(self):
        # build_graph-train vs validate-train
        print('Building the TensorFlow graph...')
        opts = self.options

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.enc_input = tf.placeholder(tf.int32, shape = [opts.batch_size, opts.max_uttr_len_enc])
            self.dec_input = tf.placeholder(tf.int32, shape = [opts.batch_size, opts.max_uttr_len_dec])
            self.target    = tf.placeholder(tf.int32, shape = [opts.batch_size, opts.max_uttr_len_dec])

            self.enc_input_len = tf.placeholder(tf.int32, shape = [opts.batch_size])
            self.dec_input_len = tf.placeholder(tf.int32, shape = [opts.batch_size])
            
            self.VAD = tf.placeholder(tf.float32, shape = [opts.corpus_size,3])
            self.termfreq = tf.placeholder(tf.float32, shape = [opts.corpus_size,1])

            with tf.variable_scope('embedding', reuse = tf.AUTO_REUSE):
                # how to get input_embed for encoder and decoder
                # word_embeddings = tf.Variable(tf.random_uniform([opts.vocab_size, opts.word_embed_size], -1.0, 1.0))
                word_embeddings = tf.Variable(opts.word_embeddings, name = 'word_embeddings')
#                 VAD = opts.VAD#, name = 'VAD')
#                 termfreq = tf.placeholder(opts.termfreq, name = 'termfreq')
                
                enc_input_embed = tf.nn.embedding_lookup(word_embeddings, self.enc_input)
                dec_input_embed = tf.nn.embedding_lookup(word_embeddings, self.dec_input)
                
                enc_input_VAD = tf.nn.embedding_lookup(self.VAD, self.enc_input)
                target_VAD = tf.nn.embedding_lookup(self.VAD, self.target)
                
                enc_input_tf = tf.nn.embedding_lookup(self.termfreq, self.enc_input)
                target_tf = tf.nn.embedding_lookup(self.termfreq, self.target)

            with tf.variable_scope('encoding', reuse = tf.AUTO_REUSE):
                cell_enc = tf.nn.rnn_cell.GRUCell(opts.n_hidden_units_enc)
                # bi-directional?
                enc_outputs, _ = tf.nn.dynamic_rnn(cell_enc, enc_input_embed,
                                                   sequence_length = self.enc_input_len,
                                                   dtype = tf.float32)

            if opts.mode == 'PREDICT':
                enc_outputs = tile_batch(enc_outputs, multiplier = opts.beam_width)
                tiled_enc_input_len = tile_batch(self.enc_input_len, multiplier = opts.beam_width)
            else:
                tiled_enc_input_len = self.enc_input_len

            with tf.variable_scope('decoding', reuse = tf.AUTO_REUSE) as vs:
                # attn_mechanism: alpha_<t,t'>
                attn_mechanism = MyBahdanauAttention(
                    num_units=opts.attn_depth,
                    memory = enc_outputs,
                    memory_sequence_length = tiled_enc_input_len,
                    enc_input_embed = enc_input_embed,
                    enc_input_VAD = enc_input_VAD,
                    enc_input_tf = enc_input_tf
                    )
                cell_dec = tf.nn.rnn_cell.GRUCell(opts.n_hidden_units_dec)
                # AttentionWrapper: c?
                cell_dec = AttentionWrapper(cell_dec, attn_mechanism, output_attention = False)
                output_layer = tf.layers.Dense(units = opts.vocab_size,
                    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1))

                # Train
                if opts.mode == 'TRAIN':
                    dec_initial_state = cell_dec.zero_state(opts.batch_size, tf.float32)
                    # 和上文算的AttentionWrapper联系？c?
                    # attention是一个词的context还是一整个encoder句子的？
                    attention = compute_attention(attn_mechanism, dec_initial_state.cell_state) #（1，256）
                    dec_initial_state = dec_initial_state.clone(attention = attention)
                    outputs_dec, _ = tf.nn.dynamic_rnn(cell = cell_dec,
                        inputs = dec_input_embed,
                        sequence_length = self.dec_input_len,
                        initial_state = dec_initial_state,
                        dtype = tf.float32,
                        scope = vs)
                    # decoder的结果？
                    # logits: vector (batch * max_utterance_len * V)
                    logits = output_layer.apply(outputs_dec)
                    # batch size * max sentence length; binary; 0 for non-word in orignal sentence
                    weights = tf.sequence_mask(self.dec_input_len,
                        maxlen = opts.max_uttr_len_dec, dtype = tf.float32)
                    delta = 0.15
                    # cross entropy loss function defined by Affective attention
#                     norm_VAD_target = 1+delta*tf.norm(target_VAD,2,2)
#                     phi = -1 * opts.corpus_size * norm_VAD_target/tf.reduce_sum(norm_VAD_target)
#                     phi = tf.matmul(phi,tf.log(target_tf))
                    self.loss = sequence_loss(logits, self.target, weights)
                    self.loss_batch = sequence_loss(logits, self.target, weights, average_across_batch = False)
                    self.optimizer = tf.train.AdamOptimizer(opts.learning_rate).minimize(self.loss)
                    self.init = tf.global_variables_initializer()

                # Predict
                if opts.mode == 'PREDICT':
                    dec_initial_state = cell_dec.zero_state(opts.batch_size * opts.beam_width, tf.float32)
                    attention = compute_attention(attn_mechanism, dec_initial_state.cell_state)
                    dec_initial_state = dec_initial_state.clone(attention = attention)
                    start_tokens = tf.constant(opts.go_index, dtype = tf.int32, shape = [opts.batch_size])
                    bs_decoder = BeamSearchDecoder(cell = cell_dec,
                        embedding = word_embeddings,
                        start_tokens = start_tokens,
                        end_token = opts.eos_index,
                        initial_state = dec_initial_state,
                        beam_width = opts.beam_width,
                        output_layer = output_layer)
                    final_outputs, final_state, _ = dynamic_decode(bs_decoder,
                        impute_finished = False,
                        maximum_iterations = opts.max_uttr_len_dec,
                        scope = vs)
                    self.predicted_ids = final_outputs.predicted_ids

            self.tvars = tf.trainable_variables()
            self.saver = tf.train.Saver(max_to_keep = 100)

    def init_tf_vars(self):
        if self.options.mode == 'TRAIN':
            self.session.run(self.init)
            print('TensorFlow variables initialized.')

    def validate(self, valid_set,VAD,termfreq):
        """Validate the model on the validation set.
        Args:
            valid_set: Dictionary containing:
                enc_input: Input to the word-level encoders. Shaped `[N, max_uttr_len]`.
                dec_input: Input to the decoder. Shaped `[N, max_uttr_len]`.
                target: Targets, expected output of the decoder. Shaped `[N, max_uttr_len]`.
                enc_input_len: Lengths of the input to the word-level encoders. Shaped `[N]`.
                dec_input_len: Lengths of the input to the decoder. Shaped `[N]`.
                (N should be a multiple of batch_size)
        Returns:
            perplexity: Perplexity on the validation set.
        """
        opts = self.options
        num_examples = valid_set['enc_input'].shape[0]
        num_batches = num_examples // opts.batch_size
        loss = 0.0
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            feed_dict = {self.enc_input: valid_set['enc_input'][s:t,:],
                         self.dec_input: valid_set['dec_input'][s:t,:],
                         self.target: valid_set['target'][s:t,:],
                         self.VAD: VAD,
                         self.termfreq: termfreq}
            loss += self.session.run(self.loss, feed_dict = feed_dict)
        return np.exp(loss / num_batches)

    def validate_batch(self, valid_setVAD,termfreq):
        feed_dict = {self.enc_input: valid_set['enc_input'],
                     self.dec_input: valid_set['dec_input'],
                     self.target: valid_set['target'],
                     self.enc_input_len: valid_set['enc_input_len'],
                     self.dec_input_len: valid_set['dec_input_len'],
                     self.VAD: VAD,
                     self.termfreq: termfreq}
        loss_batch_val = self.session.run(self.loss_batch, feed_dict = feed_dict)
        return loss_batch_val

    def train(self, train_set, VAD, termfreq, save_path, restore_epoch, valid_set = None):
        """Train the model.
        Args:
            train_set and valid_set: Dictionaries containing:
                enc_input: Input to the word-level encoders. Shaped `[N, max_uttr_len]`.
                dec_input: Input to the decoder. Shaped `[N, max_uttr_len]`.
                target: Targets, expected output of the decoder. Shaped `[N, max_uttr_len]`.
                enc_input_len: Lengths of the input to the word-level encoders. Shaped `[N]`.
                dec_input_len: Lengths of the input to the decoder. Shaped `[N]`.
        """
        print('Start to train the model...')
        opts = self.options

        num_examples = train_set['enc_input'].shape[0]
        num_batches = num_examples // opts.batch_size
        valid_ppl = [None]

        for epoch in range(opts.num_epochs):
            perm_indices = np.random.permutation(range(num_examples))
            for batch in range(num_batches):
                s = batch * opts.batch_size
                t = s + opts.batch_size
                batch_indices = perm_indices[s:t]
                feed_dict = {self.enc_input: train_set['enc_input'][batch_indices,:],
                             self.dec_input: train_set['dec_input'][batch_indices,:],
                             self.target: train_set['target'][batch_indices,:],
                             self.enc_input_len: train_set['enc_input_len'][batch_indices],
                             self.dec_input_len: train_set['dec_input_len'][batch_indices],
                             self.VAD: VAD,
                             self.termfreq: termfreq}
#                              self.termfreq: train_set['termfreq'][batch_indices]}
                _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict = feed_dict)
                print('Epoch {:03d}/{:03d}, valid ppl = {}, batch {:04d}/{:04d}, train loss = {}'.format(epoch + 1,
                    opts.num_epochs, valid_ppl[-1], batch + 1, num_batches, loss_val), flush = True)

            if valid_set is not None:
                valid_ppl.append(self.validate(valid_set,VAD, termfreq))
            self.save(os.path.join(save_path, 'model_epoch_{:03d}.ckpt'.format(restore_epoch + epoch + 1)))

        if valid_set is not None:
            for epoch in range(opts.num_epochs):
                print('Epoch {:03d}, valid ppl = {}'.format(epoch + 1, valid_ppl[epoch + 1]))

    def predict(self, enc_input, enc_input_len, VAD, termfreq):
        """Predict the response based on the input.
        Args:
            enc_input: Input to the word-level encoders. Shaped `[N, max_uttr_len]`.
            enc_input_len: Lengths of the input to the word-level encoders. Shaped `[N]`.
            (N should be a multiple of batch_size)
        Returns:
            prediction: Predicted word indices. Shaped `[N, max_uttr_len, beam_width]`.
        """
        opts = self.options
        num_examples = enc_input.shape[0]
        num_batches = num_examples // opts.batch_size
        prediction = []
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            feed_dict = {self.enc_input: enc_input[s:t,:],
                         self.enc_input_len: enc_input_len[s:t],
                         self.VAD: VAD,
                         self.termfreq: termfreq}
            p = self.session.run(self.predicted_ids, feed_dict = feed_dict)
            prediction.append(p)
        return prediction

    def save(self, save_path):
        print('Saving the trained model to {}...'.format(save_path))
        self.saver.save(self.session, save_path)

    def restore(self, restore_path):
        print('Restoring a pre-trained model from {}...'.format(restore_path))
        self.saver.restore(self.session, restore_path)
