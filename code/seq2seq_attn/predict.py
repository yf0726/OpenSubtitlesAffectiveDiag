import os
import pickle
import argparse
import numpy as np
from model import Options, Seq2SeqAttn

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = '../data/datasets/final_rf/negative_dialogs_s2s',
                    help = 'the directory to the data')
parser.add_argument('--word_embeddings_path', type = str, default = '../data/datasets/final_rf/word_embeddings_dd.npy',
                    help = 'the directory to the pre-trained word embeddings')
parser.add_argument('--num_epochs', type = int, default = 10,
                    help = 'the number of epochs to train the data')
parser.add_argument('--batch_size', type = int, default = 1,
                    help = 'the batch size')
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'the learning rate')
parser.add_argument('--beam_width', type = int, default = 256,
                    help = 'the beam width when decoding')
parser.add_argument('--word_embed_size', type = int, default = 256,
                    help = 'the size of word embeddings')
parser.add_argument('--n_hidden_units_enc', type = int, default = 256,
                    help = 'the number of hidden units of encoder')
parser.add_argument('--n_hidden_units_dec', type = int, default = 256,
                    help = 'the number of hidden units of decoder')
parser.add_argument('--attn_depth', type = int, default = 128,
                    help = 'attention depth')
parser.add_argument('--restore_path', type = str, default = 'model_dailydialog_rf',
                    help = 'the path to restore the trained model')
parser.add_argument('--restore_epoch', type = int, default = 11,
                    help = 'the epoch to restore')
args = parser.parse_args()

def read_data(data_path):
    def load_np_files(path):
        my_set = {}
        my_set['enc_input'] = np.load(os.path.join(path, 'enc_input.npy'))
        my_set['dec_input'] = np.load(os.path.join(path, 'dec_input.npy'))
        my_set['target'] = np.load(os.path.join(path, 'target.npy'))
        my_set['enc_input_len'] = np.load(os.path.join(path, 'enc_input_len.npy'))
        my_set['dec_input_len'] = np.load(os.path.join(path, 'dec_input_len.npy'))
        return my_set
    test_set = load_np_files(data_path)
    with open(os.path.join(data_path, '../token2id.pickle'), 'rb') as file:
        token2id = pickle.load(file)
    with open(os.path.join(data_path, '../id2token.pickle'), 'rb') as file:
        id2token = pickle.load(file)
    return test_set, token2id, id2token

def ids_to_sentence(ids, uttr_len, id2token):
    tokens = []
    if uttr_len is not None:
        for i in range(uttr_len):
            if id2token[ids[i]] != '<eos>' and id2token[ids[i]] != '<go>':
                tokens.append(id2token[ids[i]])
    else:
        i = 0
        while i < len(ids) and id2token[ids[i]] != '<eos>':
            tokens.append(id2token[ids[i]])
            i += 1
    return ' '.join(tokens)

if __name__ == '__main__':
    test_set, token2id, id2token = read_data(args.data_path)
    max_uttr_len_enc = test_set['enc_input'].shape[1]
    max_uttr_len_dec = test_set['dec_input'].shape[1]

    word_embeddings = np.load(args.word_embeddings_path)

    options = Options(mode = 'PREDICT',
                      num_epochs = args.num_epochs,
                      batch_size = args.batch_size,
                      learning_rate = args.learning_rate,
                      beam_width = args.beam_width,
                      vocab_size = len(token2id),
                      max_uttr_len_enc = max_uttr_len_enc,
                      max_uttr_len_dec = max_uttr_len_dec,
                      go_index = token2id['<go>'],
                      eos_index = token2id['<eos>'],
                      word_embed_size = args.word_embed_size,
                      n_hidden_units_enc = args.n_hidden_units_enc,
                      n_hidden_units_dec = args.n_hidden_units_dec,
                      attn_depth = args.attn_depth,
                      word_embeddings = word_embeddings)
    model = Seq2SeqAttn(options)

    for var in model.tvars:
        print(var.name)

    model.restore(os.path.join(args.restore_path, 'model_epoch_{:03d}.ckpt'.format(args.restore_epoch)))
    prediction = model.predict(test_set['enc_input'], test_set['enc_input_len'])

    pred_S2S = []
    f = open('neg/pred_S2S.txt', 'w', encoding = 'utf-8')
    N = (test_set['enc_input'].shape[0] // args.batch_size) * args.batch_size
    for i in range(N):
        f.write('HISTORY:\n')
        uttr = ids_to_sentence(test_set['enc_input'][i,:], test_set['enc_input_len'][i], id2token)
        f.write('- {}\n'.format(uttr))
        f.write('LABEL:\n')
        label = ids_to_sentence(test_set['target'][i,:], test_set['dec_input_len'][i], id2token)
        f.write('- {}\n'.format(label))
        f.write('PREDICTION:\n')
        pred = ids_to_sentence(prediction[i//args.batch_size][i%args.batch_size,:,0], None, id2token)
        f.write('- {}\n\n'.format(pred))
        pred_S2S.append(pred)
    f.close()
    with open('neg/pred_S2S.pickle', 'wb') as f:
        pickle.dump(pred_S2S, f)