import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.layers import Layer, Input, Dense, LSTM, Embedding, Dropout, Bidirectional, Flatten
from keras.models import Model
from keras.activations import tanh, softmax
from keras.optimizers import Adam
import logging
from Utils import load_w2v, load_inputs_twitter

logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------------------------------------------------------------------------------------

embedding_type = 'BERT'
year = 2014
embedding_dim = 768

n_lstm = 300
drop_lstm = 0.7676
beta_1 = 0.9
beta_2 = 0.999
learning_rate = 0.001
epoch = 10

max_sentence_len = 150
max_target_len = 25

# RESTAURANT
embedding_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/2014/2014BERT_emb.txt'
embedding_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/ARTS2014/2014BERT_emb.txt'
embedding_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/2014ARTS++/2014BERT_emb.txt'
train_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/2014/2014trainBERT.txt'
train_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/ARTS2014/2014trainBERT.txt'
train_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/2014ARTS++/2014trainBERT.txt'
test_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/2014/2014testBERT.txt'
test_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/ARTS2014/2014testBERT.txt'
test_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/2014ARTS++/2014testBERT.txt'

# LAPTOP
embedding_path_laptop_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/original/complete.txt'
embedding_path_laptop_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS/complete.txt'
embedding_path_laptop_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS++/complete.txt'
train_path_laptop_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/original/train.txt'
train_path_laptop_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS/train.txt'
train_path_laptop_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS++/train.txt'
test_path_laptop_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/original/test.txt'
test_path_laptop_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS/test.txt'
test_path_laptop_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS++/test.txt'

embedding_path_MAMS_test = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/MAMS/test.txt'
test_path_MAMS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/MAMS/test.txt'


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

print("[INFO] Loading word emmeddings...")
# word_id_mapping, word_embeddings_matrix = load_w2v(embedding_path_og, embedding_dim)
# word_id_mapping_ARTS, word_embeddings_matrix_ARTS = load_w2v(embedding_path_ARTS, embedding_dim)
# word_id_mapping_ARTS_PLUS_PLUS, word_embeddings_matrix_ARTS_PLUS_PLUS = load_w2v(embedding_path_ARTS_PLUS_PLUS, embedding_dim)
#
# word_id_mapping_MAMS_test, word_embeddings_matrix_MAMS_test = load_w2v(embedding_path_MAMS_test, embedding_dim)


word_id_mapping_laptop_og, word_embeddings_matrix_laptop_og = load_w2v(embedding_path_laptop_og, embedding_dim)
word_id_mapping_laptop_ARTS, word_embeddings_matrix_laptop_ARTS = load_w2v(embedding_path_laptop_ARTS, embedding_dim)
word_id_mapping_laptop_ARTS_PLUS_PLUS, word_embeddings_matrix_laptop_ARTS_PLUS_PLUS = load_w2v(
    embedding_path_laptop_ARTS_PLUS_PLUS, embedding_dim)

# vocab_size_og = word_embeddings_matrix.shape[0]
# vocab_size_ARTS = word_embeddings_matrix_ARTS.shape[0]
# vocab_size_ARTS_PLUS_PLUS = word_embeddings_matrix_ARTS.shape[0]

# vocab_size_MAMS = word_embeddings_matrix_MAMS_test.shape[0]

vocab_size_laptop_og = word_embeddings_matrix_laptop_og.shape[0]
vocab_size_laptop_ARTS = word_embeddings_matrix_laptop_ARTS.shape[0]
vocab_size_laptop_ARTS_PLUS_PLUS = word_embeddings_matrix_laptop_ARTS_PLUS_PLUS.shape[0]

print("[INFO] Loading data...")

# TRAIN
tr_x_laptop_og, tr_sen_len_laptop_og, tr_x_bw_laptop_og, tr_sen_len_bw_laptop_og, tr_y_laptop_og, tr_target_word_laptop_og, tr_tar_len_laptop_og, _, _, _ = load_inputs_twitter(
    train_path_laptop_og, word_id_mapping_laptop_og, max_sentence_len, 'TC', True, max_target_len)

tr_x_laptop_ARTS, tr_sen_len_laptop_ARTS, tr_x_bw_laptop_ARTS, tr_sen_len_bw_laptop_ARTS, tr_y_laptop_ARTS, tr_target_word_laptop_ARTS, tr_tar_len_laptop_ARTS, _, _, _ = load_inputs_twitter(
    train_path_laptop_ARTS, word_id_mapping_laptop_ARTS, max_sentence_len, 'TC', True, max_target_len)

tr_x_laptop_ARTS_PLUS_PLUS, tr_sen_len_laptop_ARTS_PLUS_PLUS, tr_x_bw_laptop_ARTS_PLUS_PLUS, tr_sen_len_bw_laptop_ARTS_PLUS_PLUS, tr_y_laptop_ARTS_PLUS_PLUS, tr_target_word_laptop_ARTS_PLUS_PLUS, tr_tar_len_laptop_ARTS_PLUS_PLUS, _, _, _ = load_inputs_twitter(
    train_path_laptop_ARTS_PLUS_PLUS, word_id_mapping_laptop_ARTS_PLUS_PLUS, max_sentence_len, 'TC', True,
    max_target_len)

# TEST
te_x_laptop_og, te_sen_len_laptop_og, te_x_bw_laptop_og, te_sen_len_bw_laptop_og, te_y_laptop_og, te_target_word_laptop_og, te_tar_len_laptop_og, _, _, _ = load_inputs_twitter(
    test_path_laptop_og, word_id_mapping_laptop_og, max_sentence_len, 'TC', True, max_target_len)

te_x_laptop_ARTS, te_sen_len_laptop_ARTS, te_x_bw_laptop_ARTS, te_sen_len_bw_laptop_ARTS, te_y_laptop_ARTS, te_target_word_laptop_ARTS, te_tar_len_laptop_ARTS, _, _, _ = load_inputs_twitter(
    test_path_laptop_ARTS, word_id_mapping_laptop_ARTS, max_sentence_len, 'TC', True, max_target_len)

te_x_laptop_ARTS_PLUS_PLUS, te_sen_len_laptop_ARTS_PLUS_PLUS, te_x_bw_laptop_ARTS_PLUS_PLUS, te_sen_len_bw_laptop_ARTS_PLUS_PLUS, te_y_laptop_ARTS_PLUS_PLUS, te_target_word_laptop_ARTS_PLUS_PLUS, te_tar_len_laptop_ARTS_PLUS_PLUS, _, _, _ = load_inputs_twitter(
    test_path_laptop_ARTS_PLUS_PLUS, word_id_mapping_laptop_ARTS_PLUS_PLUS, max_sentence_len, 'TC', True,
    max_target_len)

# trainX
trainX_laptop_og = [np.asarray(tr_x_laptop_og).astype('float32'),
                    np.asarray(tr_target_word_laptop_og).astype('float32'),
                    np.asarray(tr_x_bw_laptop_og).astype('float32')]
trainY_laptop_og = np.asarray(tr_y_laptop_og).astype('float32')

trainX_laptop_ARTS = [np.asarray(tr_x_laptop_ARTS).astype('float32') + vocab_size_laptop_og,
                      np.asarray(tr_target_word_laptop_ARTS).astype('float32') + vocab_size_laptop_og,
                      np.asarray(tr_x_bw_laptop_ARTS).astype('float32') + vocab_size_laptop_og]
trainY_laptop_ARTS = np.asarray(tr_y_laptop_ARTS).astype('float32')

trainX_laptop_ARTS_PLUS_PLUS = [np.asarray(tr_x_laptop_ARTS_PLUS_PLUS).astype('float32') + vocab_size_laptop_og + vocab_size_laptop_ARTS,
                                np.asarray(tr_target_word_laptop_ARTS_PLUS_PLUS).astype('float32') + vocab_size_laptop_og + vocab_size_laptop_ARTS,
                                np.asarray(tr_x_bw_laptop_ARTS_PLUS_PLUS).astype('float32') + vocab_size_laptop_og + vocab_size_laptop_ARTS]
trainY_laptop_ARTS_PLUS_PLUS = np.asarray(tr_y_laptop_ARTS_PLUS_PLUS).astype('float32')

# testX
testX_laptop_og = [np.asarray(te_x_laptop_og).astype('float32'),
                   np.asarray(te_target_word_laptop_og).astype('float32'),
                   np.asarray(te_x_bw_laptop_og).astype('float32')]
testY_laptop_og = np.asarray(te_y_laptop_og).astype('float32')

testX_laptop_ARTS = [np.asarray(te_x_laptop_ARTS).astype('float32') + vocab_size_laptop_og,
                     np.asarray(te_target_word_laptop_ARTS).astype('float32') + vocab_size_laptop_og,
                     np.asarray(te_x_bw_laptop_ARTS).astype('float32') + vocab_size_laptop_og]
testY_laptop_ARTS = np.asarray(te_y_laptop_ARTS).astype('float32')

testX_laptop_ARTS_PLUS_PLUS = [np.asarray(te_x_laptop_ARTS_PLUS_PLUS).astype('float32') + vocab_size_laptop_og + vocab_size_laptop_ARTS,
                               np.asarray(te_target_word_laptop_ARTS_PLUS_PLUS).astype('float32') + vocab_size_laptop_og + vocab_size_laptop_ARTS,
                               np.asarray(te_x_bw_laptop_ARTS_PLUS_PLUS).astype('float32') + vocab_size_laptop_og + vocab_size_laptop_ARTS]
testY_laptop_ARTS_PLUS_PLUS = np.asarray(te_y_laptop_ARTS_PLUS_PLUS).astype('float32')

input_layer_l = Input(shape=(max_sentence_len,))
embedding_layer_l = Embedding(input_dim=vocab_size_laptop_og + vocab_size_laptop_ARTS + vocab_size_laptop_ARTS_PLUS_PLUS,
                              output_dim=embedding_dim,
                              weights=[np.vstack((word_embeddings_matrix_laptop_og,
                                                  word_embeddings_matrix_laptop_ARTS,
                                                  word_embeddings_matrix_laptop_ARTS_PLUS_PLUS))],
                              trainable=False)(input_layer_l)
h_l = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_l)
h_l = Dropout(drop_lstm)(h_l)

input_layer_c = Input(shape=(max_target_len,))
embedding_layer_c = Embedding(input_dim=vocab_size_laptop_og + vocab_size_laptop_ARTS + vocab_size_laptop_ARTS_PLUS_PLUS,
                              output_dim=embedding_dim,
                              weights=[np.vstack((word_embeddings_matrix_laptop_og,
                                                  word_embeddings_matrix_laptop_ARTS,
                                                  word_embeddings_matrix_laptop_ARTS_PLUS_PLUS))],
                              trainable=False)(input_layer_c)
h_c = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_c)
h_c = Dropout(drop_lstm)(h_c)

input_layer_r = Input(shape=(max_sentence_len,))
embedding_layer_r = Embedding(input_dim=vocab_size_laptop_og + vocab_size_laptop_ARTS + vocab_size_laptop_ARTS_PLUS_PLUS,
                              output_dim=embedding_dim,
                              weights=[np.vstack((word_embeddings_matrix_laptop_og,
                                                  word_embeddings_matrix_laptop_ARTS,
                                                  word_embeddings_matrix_laptop_ARTS_PLUS_PLUS))],
                              trainable=False)(input_layer_r)
h_r = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_r)
h_r = Dropout(drop_lstm)(h_r)

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

print("[INFO] Training model...")

print("Running model with drop_lstm = " + str(drop_lstm) + " and number of epochs: " + str(epoch))
model.fit(x=trainX_laptop_ARTS, y=trainY_laptop_ARTS,
          validation_data=(testX_laptop_ARTS, testY_laptop_ARTS),
          epochs=epoch, batch_size=20, shuffle=True)

preds_laptop_og = model.predict(testX_laptop_og)
preds_laptop_ARTS = model.predict(testX_laptop_ARTS)
preds_laptop_ARTS_PLUS_PLUS = model.predict(testX_laptop_ARTS_PLUS_PLUS)

correct_pred_laptop_og = np.equal(np.argmax(preds_laptop_og, 1), np.argmax(te_y_laptop_og, 1))
correct_pred_laptop_ARTS = np.equal(np.argmax(preds_laptop_ARTS, 1), np.argmax(te_y_laptop_ARTS, 1))
correct_pred_laptop_ARTS_PLUS_PLUS = np.equal(np.argmax(preds_laptop_ARTS_PLUS_PLUS, 1), np.argmax(te_y_laptop_ARTS_PLUS_PLUS, 1))

test_acc_prob_laptop_og = np.mean(correct_pred_laptop_og)
test_acc_prob_laptop_ARTS = np.mean(correct_pred_laptop_ARTS)
test_acc_prob_laptop_ARTS_PLUS_PLUS = np.mean(correct_pred_laptop_ARTS_PLUS_PLUS)


model.save(f"data/results/2014/Laptop/ARTS/Model_{test_acc_prob_laptop_ARTS:.1%}")

with open("data/neural_preds/laptop/ARTS_trained/original.txt", "w") as file:
    for item in correct_pred_laptop_og:
        file.write(str(item) + "\n")
file.close()

with open("data/neural_preds/laptop/ARTS_trained/ARTS.txt", "w") as file:
    for item in correct_pred_laptop_ARTS:
        file.write(str(item) + "\n")
file.close()

with open("data/neural_preds/laptop/ARTS_trained/ARTS++.txt", "w") as file:
    for item in correct_pred_laptop_ARTS_PLUS_PLUS:
        file.write(str(item) + "\n")
file.close()

print(f"Test Accuracy og: {test_acc_prob_laptop_og:.1%}")
print(f"Test Accuracy ARTS: {test_acc_prob_laptop_ARTS:.1%}")
print(f"Test Accuracy ARTS++: {test_acc_prob_laptop_ARTS_PLUS_PLUS:.1%}")


