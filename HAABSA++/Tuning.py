import random
import json
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.layers import Layer, Input, Dense, LSTM, Embedding, Dropout, Bidirectional, Flatten
from keras.models import Model
from keras.activations import tanh, softmax
from keras.optimizers import Adam
import logging
from hyperopt import tpe, hp, STATUS_OK, Trials, fmin

logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------------------------------------------------------------------------------------

embedding_type = 'BERT'
year = 2014
embedding_dim = 768

n_lstm = 300

space = {
    'dropout_rate': hp.uniform('dropout_rate', 0.6, 0.9),
    'beta_1': hp.uniform('beta_1', 0.8, 0.99),
    'beta_2': hp.uniform('beta_2', 0.99, 0.9999),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.00001)
}

max_sentence_len = 150
max_target_len = 25

# embedding_path_laptop_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/original/complete.txt'
# embedding_path_laptop_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS/complete.txt'
embedding_path_laptop_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS++/complete.txt'
# embedding_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/original/2014BERT_emb.txt'
# embedding_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS/2014BERT_emb.txt'
# embedding_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS++/2014BERT_emb.txt'
# train_path_laptop_og ='/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/original/train.txt'
# train_path_laptop_ARTS ='/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS/train.txt'
train_path_laptop_ARTS_PLUS_PLUS ='/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS++/train.txt'
# train_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/original/2014trainBERT.txt'
# train_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS/2014trainBERT.txt'
# train_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS++/2014trainBERT.txt'


# -------------------------------------------------------------------------------------------------

class BilinearAttentionLayer(Layer):
    def __init__(self, use_bias=True, **kwargs):
        super(BilinearAttentionLayer, self).__init__(**kwargs)
        self.use_bias = use_bias

    def build(self, input_shape):
        # Create trainable weight matrix W
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1][-1], input_shape[1][-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        if self.use_bias:
            # Create trainable bias vector b
            self.b = self.add_weight(name='b',
                                     shape=(1),
                                     initializer='zeros',
                                     trainable=True)
        super(BilinearAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        input1, input2 = inputs
        # Perform the bilinear operation
        result = tf.matmul(input1, tf.transpose(self.W))
        result = tf.reduce_sum(result * input2, axis=-1, keepdims=True)
        if self.use_bias:
            result += self.b  # Add bias term if specified
        return softmax(tanh(result), axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0]


# -------------------------------------------------------------------------------------------------

def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    word_dict['$t$'] = (cnt + 1)
    return word_dict, w2v


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    # print('\nload word-id mapping done!\n')
    return word_to_id


def change_y_to_onehot(y):
    class_set = sorted(set(y))
    n_class = len(class_set)
    y_onehot_mapping = {}
    for index, label in enumerate(class_set):
        y_onehot_mapping[label] = index
    print("HERE IS THE DICTIONARY: ", y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    # print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    # read in txt file
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        # targets
        words = lines[i + 1].lower().split()
        target = words

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # sentiment
        y.append(lines[i + 2].strip().split()[0])

        # left and right context
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y
    y = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(x_r, dtype="object"), \
            np.asarray(sen_len_r, dtype="object"), np.asarray(y, dtype="object")
    elif type_ == 'TC':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(x_r,
                                                                                              dtype="object"), np.asarray(
            sen_len_r, dtype="object"), \
            np.asarray(y, dtype="object"), np.asarray(target_words, dtype="object"), np.asarray(tar_len,
                                                                                                dtype="object"), np.asarray(
            all_sent, dtype="object"), np.asarray(all_target, dtype="object"), np.asarray(all_y, dtype="object")
    elif type_ == 'IAN':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(target_words,
                                                                                              dtype="object"), \
            np.asarray(tar_len, dtype="object"), np.asarray(y, dtype="object")
    else:
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(y, dtype="object")


# -------------------------------------------------------------------------------------------------
print("[INFO] Loading word embeddings...")
# word_id_mapping, word_embeddings_matrix = load_w2v(embedding_path_og, embedding_dim)

# word_id_mapping_laptop_og, word_embeddings_matrix_laptop_og = load_w2v(embedding_path_laptop_og, embedding_dim)

# word_id_mapping_laptop_ARTS, word_embeddings_matrix_laptop_ARTS = load_w2v(embedding_path_laptop_ARTS, embedding_dim)
word_id_mapping_laptop_ARTS_PLUS_PLUS, word_embeddings_matrix_laptop_ARTS_PLUS_PLUS = load_w2v(embedding_path_laptop_ARTS_PLUS_PLUS, embedding_dim)

# word_id_mapping_ARTS_PLUS_PLUS, word_embeddings_matrix_ARTS_PLUS_PLUS = load_w2v(embedding_path_ARTS_PLUS_PLUS,
#                                                                                  embedding_dim)

# vocab_size_og = word_embeddings_matrix.shape[0]
vocab_size_laptop_ARTS_PLUS_PLUS = word_embeddings_matrix_laptop_ARTS_PLUS_PLUS.shape[0]
# vocab_size_ARTS_PLUS_PLUS = word_embeddings_matrix_ARTS_PLUS_PLUS.shape[0]


def HAABSA_model(learning_rate, beta_1, beta_2, dropout_rate):
    input_layer_l = Input(shape=(max_sentence_len,))
    embedding_layer_l = Embedding(input_dim=vocab_size_laptop_ARTS_PLUS_PLUS,
                                  output_dim=embedding_dim,
                                  weights=[word_embeddings_matrix_laptop_ARTS_PLUS_PLUS],
                                  trainable=False)(input_layer_l)
    h_l = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_l)
    h_l = Dropout(dropout_rate)(h_l)

    input_layer_c = Input(shape=(max_target_len,))
    embedding_layer_c = Embedding(input_dim=vocab_size_laptop_ARTS_PLUS_PLUS,
                                  output_dim=embedding_dim,
                                  weights=[word_embeddings_matrix_laptop_ARTS_PLUS_PLUS],
                                  trainable=False)(input_layer_c)
    h_c = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_c)
    h_c = Dropout(dropout_rate)(h_c)

    input_layer_r = Input(shape=(max_sentence_len,))
    embedding_layer_r = Embedding(input_dim=vocab_size_laptop_ARTS_PLUS_PLUS,
                                  output_dim=embedding_dim,
                                  weights=[word_embeddings_matrix_laptop_ARTS_PLUS_PLUS],
                                  trainable=False)(input_layer_r)
    h_r = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_r)
    h_r = Dropout(dropout_rate)(h_r)

    BAL_l = BilinearAttentionLayer(use_bias=True)
    BAL_r = BilinearAttentionLayer(use_bias=True)
    BAL_cl = BilinearAttentionLayer(use_bias=True)
    BAL_cr = BilinearAttentionLayer(use_bias=True)

    HAL_c = Dense(units=1, activation='tanh')
    HAL_t = Dense(units=1, activation='tanh')

    for i in range(3):
        if i == 0:
            r_cp = tf.reduce_mean(h_c, axis=1, keepdims=True)

            alfa_l = BAL_l([h_l, r_cp])
            r_l = tf.matmul(alfa_l, h_l, transpose_a=True)
            alfa_r = BAL_r([h_r, r_cp])
            r_r = tf.matmul(alfa_r, h_r, transpose_a=True)
            alfa_cl = BAL_cl([h_c, r_l])
            r_cl = tf.matmul(alfa_cl, h_c, transpose_a=True)
            alfa_cr = BAL_cr([h_c, r_r])
            r_cr = tf.matmul(alfa_cr, h_c, transpose_a=True)

            h_alfa_l, h_alfa_r = tf.split(softmax(tf.concat([HAL_c(r_l), HAL_c(r_r)], axis=1), axis=1),
                                          num_or_size_splits=2, axis=1)
            r_l = tf.matmul(h_alfa_l, r_l)
            r_r = tf.matmul(h_alfa_r, r_r)
            h_alfa_cl, h_alfa_cr = tf.split(softmax(tf.concat([HAL_t(r_cl), HAL_t(r_cr)], axis=1), axis=1),
                                            num_or_size_splits=2, axis=1)
            r_cl = tf.matmul(h_alfa_cl, r_cl)
            r_cr = tf.matmul(h_alfa_cr, r_cr)

        alfa_l = BAL_l([h_l, r_cl])
        r_l = tf.matmul(alfa_l, h_l, transpose_a=True)
        alfa_r = BAL_r([h_r, r_cr])
        r_r = tf.matmul(alfa_r, h_r, transpose_a=True)
        alfa_cl = BAL_cl([h_c, r_l])
        r_cl = tf.matmul(alfa_cl, h_c, transpose_a=True)
        alfa_cr = BAL_cr([h_c, r_r])
        r_cr = tf.matmul(alfa_cr, h_c, transpose_a=True)

        h_alfa_l, h_alfa_r = tf.split(softmax(tf.concat([HAL_c(r_l), HAL_c(r_r)], axis=1), axis=1),
                                      num_or_size_splits=2,
                                      axis=1)
        r_l = tf.matmul(h_alfa_l, r_l)
        r_r = tf.matmul(h_alfa_r, r_r)
        h_alfa_cl, h_alfa_cr = tf.split(softmax(tf.concat([HAL_t(r_cl), HAL_t(r_cr)], axis=1), axis=1),
                                        num_or_size_splits=2, axis=1)
        r_cl = tf.matmul(h_alfa_cl, r_cl)
        r_cr = tf.matmul(h_alfa_cr, r_cr)

    v = tf.concat([r_l, r_cl, r_cr, r_r], axis=2)

    p = Dense(units=3, activation='softmax')(v)
    p = Flatten()(p)

    model = Model(inputs=[input_layer_l, input_layer_c, input_layer_r], outputs=p)
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    return model


print("[INFO] Loading data...")
tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
    train_path_laptop_ARTS_PLUS_PLUS, word_id_mapping_laptop_ARTS_PLUS_PLUS, max_sentence_len, 'TC', True, max_target_len)
indices = list(range(0, len(tr_x)))

len_test = int(0.2 * len(tr_x))
test_indices = sorted(random.sample(indices, len_test))
train_indices = list(set(indices) - set(test_indices))

hyper_tr_x = tr_x[train_indices]
hyper_tr_target_word = tr_target_word[train_indices]
hyper_tr_x_bw = tr_x_bw[train_indices]
hyper_tr_y = tr_y[train_indices]

hyper_te_x = tr_x[test_indices]
hyper_te_target_word = tr_target_word[test_indices]
hyper_te_x_bw = tr_x_bw[test_indices]
hyper_te_y = tr_y[test_indices]

trainX = [np.asarray(hyper_tr_x).astype('float32'), np.asarray(hyper_tr_target_word).astype('float32'),
          np.asarray(hyper_tr_x_bw).astype('float32')]
trainY = np.asarray(hyper_tr_y).astype('float32')

testX = [np.asarray(hyper_te_x).astype('float32'), np.asarray(hyper_te_target_word).astype('float32'),
         np.asarray(hyper_te_x_bw).astype('float32')]
testY = np.asarray(hyper_te_y).astype('float32')


def objective(params):
    params['dropout_rate'] = float(params['dropout_rate'])
    params['learning_rate'] = float(params['learning_rate'])
    params['beta_1'] = float(params['beta_1'])
    params['beta_2'] = float(params['beta_2'])

    model = HAABSA_model(params['learning_rate'], params['beta_1'], params['beta_2'], params['dropout_rate'])
    model.fit(x=trainX, y=trainY,
              validation_data=(testX, testY),
              epochs=10, batch_size=20, shuffle=True)
    preds = model.predict(testX)
    correct_pred = np.equal(np.argmax(preds, 1), np.argmax(hyper_te_y, 1))
    test_acc_prob = np.mean(correct_pred)
    return {'loss': 1 - test_acc_prob, 'status': STATUS_OK}


trials = Trials()

best_hyperparameters = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=10,
                            trials=trials)

with open('data/hyperparameters/Laptop/ARTS++/best_hyperparameters.txt', 'w') as file:
    json.dump(best_hyperparameters, file)
