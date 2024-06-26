import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
import logging
import numpy as np
from Utils import load_w2v, load_inputs_twitter

# -------------------------------------------------------------------------------------------------

logging.getLogger('tensorflow').setLevel(logging.ERROR)
embedding_type = 'BERT'
year = 2014
embedding_dim = 768

n_lstm = 300

max_sentence_len = 150
max_target_len = 25

# embedding_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/2014/2014BERT_emb.txt'
# embedding_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/ARTS2014/2014BERT_emb.txt'
# embedding_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/2014ARTS++/2014BERT_emb.txt'

# test_path_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/2014/2014testBERT.txt'
# test_path_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/ARTS2014/2014testBERT.txt'
# test_path_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/2014ARTS++/2014testBERT.txt'

# embedding_path_MAMS_test = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/MAMS/test.txt'
# test_path_MAMS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/MAMS/test.txt'


# LAPTOP:
embedding_path_laptop_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/original/complete.txt'
embedding_path_laptop_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS/complete.txt'
embedding_path_laptop_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS++/complete.txt'

test_path_laptop_og = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/original/test.txt'
test_path_laptop_ARTS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS/test.txt'
test_path_laptop_ARTS_PLUS_PLUS = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS++/test.txt'

# -------------------------------------------------------------------------------------------------

print("[INFO] Loading word emmeddings...")
# word_id_mapping, word_embeddings_matrix = load_w2v(embedding_path_og, embedding_dim)
# word_id_mapping_ARTS, word_embeddings_matrix_ARTS = load_w2v(embedding_path_ARTS, embedding_dim)
# word_id_mapping_ARTS_PLUS_PLUS, word_embeddings_matrix_ARTS_PLUS_PLUS = load_w2v(embedding_path_ARTS_PLUS_PLUS, embedding_dim)

# word_id_mapping_MAMS_test, word_embeddings_matrix_MAMS_test = load_w2v(embedding_path_MAMS_test, embedding_dim)

word_id_mapping_laptop_og, word_embeddings_matrix_laptop_og = load_w2v(embedding_path_laptop_og, embedding_dim)
word_id_mapping_laptop_ARTS, word_embeddings_matrix_laptop_ARTS = load_w2v(embedding_path_laptop_ARTS, embedding_dim)
word_id_mapping_laptop_ARTS_PLUS_PLUS, word_embeddings_matrix_laptop_ARTS_PLUS_PLUS = load_w2v(
    embedding_path_laptop_ARTS_PLUS_PLUS, embedding_dim)

print("[INFO] Loading model and data...")

model = load_model(
    "/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/results/models/tuning/Laptop/ARTS/Best_model_new")

te_x_laptop_og, te_sen_len_laptop_og, te_x_bw_laptop_og, te_sen_len_bw_laptop_og, te_y_laptop_og, te_target_word_laptop_og, te_tar_len_laptop_og, _, _, _ = load_inputs_twitter(
    test_path_laptop_og, word_id_mapping_laptop_og, max_sentence_len, 'TC', True, max_target_len)

te_x_laptop_ARTS, te_sen_len_laptop_ARTS, te_x_bw_laptop_ARTS, te_sen_len_bw_laptop_ARTS, te_y_laptop_ARTS, te_target_word_laptop_ARTS, te_tar_len_laptop_ARTS, _, _, _ = load_inputs_twitter(
    test_path_laptop_ARTS, word_id_mapping_laptop_ARTS, max_sentence_len, 'TC', True, max_target_len)

te_x_laptop_ARTS_PLUS_PLUS, te_sen_len_laptop_ARTS_PLUS_PLUS, te_x_bw_laptop_ARTS_PLUS_PLUS, te_sen_len_bw_laptop_ARTS_PLUS_PLUS, te_y_laptop_ARTS_PLUS_PLUS, te_target_word_laptop_ARTS_PLUS_PLUS, te_tar_len_laptop_ARTS_PLUS_PLUS, _, _, _ = load_inputs_twitter(
    test_path_laptop_ARTS_PLUS_PLUS, word_id_mapping_laptop_ARTS_PLUS_PLUS, max_sentence_len, 'TC', True,
    max_target_len)

testX_laptop_og = [np.asarray(te_x_laptop_og).astype('float32'),
                   np.asarray(te_target_word_laptop_og).astype('float32'),
                   np.asarray(te_x_bw_laptop_og).astype('float32')]

testX_laptop_ARTS = [np.asarray(te_x_laptop_ARTS).astype('float32'),
                     np.asarray(te_target_word_laptop_ARTS).astype('float32'),
                     np.asarray(te_x_bw_laptop_ARTS).astype('float32')]

testX_laptop_ARTS_PLUS_PLUS = [np.asarray(te_x_laptop_ARTS_PLUS_PLUS).astype('float32'),
                               np.asarray(te_target_word_laptop_ARTS_PLUS_PLUS).astype('float32'),
                               np.asarray(te_x_bw_laptop_ARTS_PLUS_PLUS).astype('float32')]

print("[INFO] Predicting OG...")

preds_laptop_og = model.predict(testX_laptop_og)

current_embedding_weights = model.layers[2].get_weights()[0]

current_embedding_weights[:len(word_embeddings_matrix_laptop_ARTS)] = word_embeddings_matrix_laptop_ARTS

new_embedding_weights = [current_embedding_weights]

model.layers[2].set_weights(new_embedding_weights)
model.layers[4].set_weights(new_embedding_weights)
model.layers[6].set_weights(new_embedding_weights)

print("[INFO] Predicting ARTS...")

preds_laptop_ARTS = model.predict(testX_laptop_ARTS)

current_embedding_weights[:len(word_embeddings_matrix_laptop_ARTS_PLUS_PLUS)] = word_embeddings_matrix_laptop_ARTS_PLUS_PLUS

new_embedding_weights = [current_embedding_weights]

model.layers[2].set_weights(new_embedding_weights)
model.layers[4].set_weights(new_embedding_weights)
model.layers[6].set_weights(new_embedding_weights)

print("[INFO] Predicting ARTS++...")

preds_laptop_ARTS_PLUS_PLUS = model.predict(testX_laptop_ARTS_PLUS_PLUS)

correct_pred_laptop_og = np.equal(np.argmax(preds_laptop_og, 1), np.argmax(te_y_laptop_og, 1))
correct_pred_laptop_ARTS = np.equal(np.argmax(preds_laptop_ARTS, 1), np.argmax(te_y_laptop_ARTS, 1))
correct_pred_laptop_ARTS_PLUS_PLUS = np.equal(np.argmax(preds_laptop_ARTS_PLUS_PLUS, 1),
                                              np.argmax(te_y_laptop_ARTS_PLUS_PLUS, 1))

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

test_acc_prob_laptop_og = np.mean(correct_pred_laptop_og)
test_acc_prob_laptop_ARTS = np.mean(correct_pred_laptop_ARTS)
test_acc_prob_laptop_ARTS_PLUS_PLUS = np.mean(correct_pred_laptop_ARTS_PLUS_PLUS)

print(f"Test Accuracy og: {test_acc_prob_laptop_og:.1%}")
print(f"Test Accuracy ARTS: {test_acc_prob_laptop_ARTS:.1%}")
print(f"Test Accuracy ARTS++: {test_acc_prob_laptop_ARTS_PLUS_PLUS:.1%}")
