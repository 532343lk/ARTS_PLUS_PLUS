# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
import json
import random
import string
import language_tool_python
from nltk.tree import ParentedTree
from nltk.corpus import wordnet
from pywsd import simple_lesk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np
from utils import Utils, load_w2v, load_inputs_twitter_test
from gramformer import Gramformer
from Sentiment_prediction import process_sentence, find_target_indices
from tensorflow.keras.models import load_model

random.seed(1016)

embedding_dim = 768
n_lstm = 300
drop_lstm = 0.7

max_sentence_len = 150
max_target_len = 25

embedding_path = 'data/text files/token_embeddings.txt'
test_path = 'data/text files/modified_sentence.txt'

model = load_model('/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/results/models/tuning/Laptop/original/Best_model_new')
current_embedding_weights = model.layers[2].get_weights()[0]

# read the json file where common terms are linked to their corresponding category.
# This file is retrieved from the 2016 semeval restaurant data. "wine" is linked to "DRINKS" category for example
# with open('target_category_dict.json', 'r') as f:
#     target_category_dict = json.load(f)
# with open('categories_targets_dict.json', 'r') as f:
#     categories_targets_dict = json.load(f)
target_category_dict = {}
categories_targets_dict = {"FEATURES": "", "PERFORMANCE": "", "DESIGN": "", "USABILITY": ""}

# HERE WE MUST DEFINE OUR OWN CATEGORIES
# food_synset = wordnet.synsets('food')[0]
# restaurant_synset = wordnet.synsets('restaurant')[0]
# ambience_synset = wordnet.synsets('ambience')[1]
# service_synset = wordnet.synsets('service')[14]
# drinks_synset = wordnet.synsets('drinks')[2]
# location_synset = wordnet.synsets('location')[0]
# categories_synsets_dict = {"FOOD": food_synset, "RESTAURANT": restaurant_synset, "AMBIENCE": ambience_synset,
#                            "SERVICE": service_synset, "DRINKS": drinks_synset, "LOCATION": location_synset}


features_synset = wordnet.synsets('features')[0]
performance_synset = wordnet.synsets('performance')[4]
design_synset = wordnet.synsets('design')[1]
usability_synset = wordnet.synsets('usability')[0]
categories_synsets_dict = {"FEATURES": features_synset, "PERFORMANCE": performance_synset, "DESIGN": design_synset,
                           "USABILITY": usability_synset}


def pywsd_to_nltk(synset):
    correct_synset = None
    possible_nltk_synsets = wordnet.synsets(synset.lemmas()[0].name())
    for synset_i in possible_nltk_synsets:
        if synset_i.definition() == synset.definition():
            correct_synset = synset_i
    return correct_synset


def get_term_category(term, sentence):
    if term.lower() in target_category_dict:
        term_category = target_category_dict[term.lower()]
    else:
        # first we find the correct sense of the term. We know that the term is a noun
        term_synset = simple_lesk(sentence, term, 'n')
        possible_synsets = []
        if term_synset is None:
            if " " in term:
                tokenized_words = word_tokenize(sentence)
                tagged_words = pos_tag(tokenized_words)
                split_term = term.split(" ")
                for word, tag in tagged_words:
                    if word in split_term:
                        if tag == 'NN':
                            synset_to_append = simple_lesk(sentence, word, 'n')
                            if synset_to_append is not None:
                                possible_synsets.append(simple_lesk(sentence, word, 'n'))
        else:
            possible_synsets.append(term_synset)
        # if simple_lesk wasn't able to find any synset for the given target.
        # Even after splitting it into multiple words. We randomly allocate the
        # target to 1 of the 6 possible categories.
        if len(possible_synsets) == 0:
            term_category = random.choice(list(categories_targets_dict.keys()))
        else:
            term_category = random.choice(list(categories_targets_dict.keys()))
            max_wup_similarity = 0
            for category_name, category_synset in categories_synsets_dict.items():
                for synset in possible_synsets:
                    nltk_synset = pywsd_to_nltk(synset)
                    if nltk_synset is None:
                        continue
                    else:
                        wup_similarity = category_synset.wup_similarity(nltk_synset)
                        if wup_similarity > max_wup_similarity:
                            max_wup_similarity = wup_similarity
                            term_category = category_name
    return term_category


def revTgt(dataset, input_file, outfile):
    gf = Gramformer(models=1)
    util = Utils(dataset)
    with open(input_file, 'r', encoding='utf-8') as fr:
        lines = json.load(fr)
    res = {}
    for sid in lines:
        sent_example = lines[sid]
        sentence = sent_example['sentence']
        copy_sent = sentence
        # splits the sentence into words
        words_list = util.tokenize(copy_sent)
        # gives a dictionary where key = aspect, and the information like corresponding opinion words is stored in the dict
        term_to_position_list = util.tokenize_term_list(copy_sent, sent_example)
        for tid in term_to_position_list:
            added_not = False
            term = term_to_position_list[tid]['term']
            term_from = term_to_position_list[tid]['from']
            term_to = term_to_position_list[tid]['to']
            polarity = term_to_position_list[tid]['polarity']
            opinions = term_to_position_list[tid]['opinions']
            other_polarity = set()
            other_opinions = set()
            for tid2 in term_to_position_list:
                if tid2 != tid:
                    other_polarity.add(term_to_position_list[tid2]['polarity'])
                    for other_opi in term_to_position_list[tid2]['opinions']:
                        other_opinions.add(other_opi[0])

            cur_opinions = set()
            for cur_opi in term_to_position_list[tid]['opinions']:
                cur_opinions.add(cur_opi[0])

            if polarity == 'positive':
                new_words, new_opi_words, added_not = util.reverse(words_list, opinions, copy_sent)
                new_polarity = 'negative'
            elif polarity == 'negative':
                new_words, new_opi_words, added_not = util.reverse(words_list, opinions, copy_sent)
                new_polarity = 'positive'
            else:
                new_words1, new_opi_words1, added_not_1 = util.reverse(words_list, opinions, copy_sent)
                new_words2, new_opi_words2, added_not_2 = util.reverse(words_list, opinions, copy_sent)
                if random.random() < 0.5:
                    new_words = new_words1
                    added_not = added_not_1
                else:
                    new_words = new_words2
                    added_not = added_not_2
                new_polarity = 'neutral'

            and_ind = []
            but_ind = []
            if len(other_polarity) > 0:
                for i, w in enumerate(new_words):
                    if w.lower() in ['and', 'but']:
                        if new_polarity not in other_polarity and w.lower() == 'and' and len(
                                cur_opinions & other_opinions) == 0 and w.lower() not in term.lower():
                            and_ind.append(i)
                        elif new_polarity in other_polarity and w.lower() == 'but' and len(
                                cur_opinions & other_opinions) == 0 and w.lower() not in term.lower():
                            but_ind.append(i)

            min = 1e8
            aidx = -1
            if new_polarity not in other_polarity and len(but_ind) == 0 and len(
                    cur_opinions & other_opinions) == 0:
                for idx in and_ind:
                    if idx > term_to and idx < len(new_words) - 3:
                        if idx - term_to < min:
                            min = idx - term_to
                            aidx = idx
                    if idx < term_from and idx > 3:
                        if term_to - idx < min:
                            min = term_to - idx
                            aidx = idx
                if aidx != -1:
                    new_words[aidx] = 'but'
            elif new_polarity in other_polarity and len(and_ind) == 0 and len(
                    cur_opinions & other_opinions) == 0:
                for idx in and_ind:
                    if idx > term_to and idx < len(new_words) - 3:
                        if idx - term_to < min:
                            min = idx - term_to
                            aidx = idx
                    if idx < term_from and idx > 3:
                        if term_to - idx < min:
                            min = term_to - idx
                            aidx = idx
                if aidx != -1:
                    new_words[aidx] = 'and'

            if new_words == words_list:
                continue

            new_sent = util.untokenize(new_words)
            if sentence[0].isupper():
                new_sent = new_sent[0].upper() + new_sent[1:]

            a = ''.join(util.tokenize(new_sent))
            b = ''.join(util.tokenize(term))
            c = 0
            for dd in range(len(a)):
                if a[dd:dd + len(b)] == b:
                    c = len(a[:dd])
                    break

            span_from = 0
            c2 = 0
            for dd in range(len(new_sent)):
                if new_sent[dd] != ' ':
                    c2 += 1
                if c2 == c and c != 0 and new_sent[dd + 1] != ' ':
                    span_from = dd + 1
                    break
                if c2 == c and c != 0 and new_sent[dd + 1] == ' ':
                    span_from = dd + 2
                    break

            span_to = span_from + len(term)

            if new_sent[span_from:span_to] != term:
                print(tid, term, new_sent[span_from:span_to])

            if sentence[0].isupper():
                new_sent = new_sent[0].upper() + new_sent[1:]

            if added_not:
                # process_sentence updates the txt files modified_sentence and embedding_test to
                # contain the info for the new sentence
                corrections = gf.correct(new_sent)
                corrected_sent = next(iter(corrections), None)

                if " " + term + " " not in corrected_sent:
                    pass
                else:
                    if corrected_sent == new_sent:
                        pass
                    else:
                        process_sentence(new_sent, term)

                        word_id_mapping, word_embeddings_matrix = load_w2v(embedding_path, embedding_dim)

                        current_embedding_weights[:len(word_embeddings_matrix)] = word_embeddings_matrix

                        new_embedding_weights = [current_embedding_weights]

                        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_target_word, te_tar_len, _, _ = load_inputs_twitter_test(
                            test_path, word_id_mapping, max_sentence_len, 'TC', True, max_target_len)

                        testX = [np.asarray(te_x).astype('float32'), np.asarray(te_target_word).astype('float32'),
                                 np.asarray(te_x_bw).astype('float32')]

                        # add current sentence embeddings to the embedding layers
                        model.layers[2].set_weights(new_embedding_weights)
                        model.layers[4].set_weights(new_embedding_weights)
                        model.layers[6].set_weights(new_embedding_weights)

                        preds_og_sentence = model.predict(testX)

                        sentiment_translation_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

                        # If HAABSA makes wrong prediction for og sentence, we skip this sentence
                        if np.argmax(preds_og_sentence) != sentiment_translation_dict[new_polarity]:
                            pass
                        else:
                            process_sentence(corrected_sent, term)

                            word_id_mapping, word_embeddings_matrix = load_w2v(embedding_path, embedding_dim)

                            current_embedding_weights[:len(word_embeddings_matrix)] = word_embeddings_matrix

                            new_embedding_weights = [current_embedding_weights]

                            te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_target_word, te_tar_len, _, _ = load_inputs_twitter_test(
                                test_path, word_id_mapping, max_sentence_len, 'TC', True, max_target_len)

                            testX = [np.asarray(te_x).astype('float32'), np.asarray(te_target_word).astype('float32'),
                                     np.asarray(te_x_bw).astype('float32')]

                            # add current sentence embeddings to the embedding layers
                            model.layers[2].set_weights(new_embedding_weights)
                            model.layers[4].set_weights(new_embedding_weights)
                            model.layers[6].set_weights(new_embedding_weights)

                            preds_corrected_sentence = model.predict(testX)

                            equal_pred = np.equal(np.argmax(preds_og_sentence, 1),
                                                  np.argmax(preds_corrected_sentence, 1))

                            if equal_pred:
                                new_sent = corrected_sent
                                span_from, span_to = find_target_indices(new_sent, term)

            print(new_sent)
            res[tid] = {
                'term': term,
                'id': tid,
                'sentence': new_sent,
                'multi': sent_example['multi'],
                'contra': sent_example['contra'],
                'from': span_from,
                'to': span_to,
                'polarity': new_polarity
            }
    with open(outfile, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)


def revNon(dataset, input_file, outfile):
    util = Utils(dataset)
    with open(input_file, 'r', encoding='utf-8') as fr:
        lines = json.load(fr)

    res = {}
    for sid in lines:
        sent_example = lines[sid]
        sentence = sent_example['sentence']
        term_list = sent_example['term_list']

        # do in multiple terms
        if len(term_list) == 1:
            continue

        copy_sent = sentence
        words_list = util.tokenize(copy_sent)
        term_to_position_list = util.tokenize_term_list(copy_sent, sent_example)

        all_id = []
        all_polarity = []
        all_opinions = []
        for tid in term_to_position_list:
            added_not = False
            term = term_to_position_list[tid]['term']
            term_from = term_to_position_list[tid]['from']
            term_to = term_to_position_list[tid]['to']
            polarity = term_to_position_list[tid]['polarity']
            opinions = term_to_position_list[tid]['opinions']
            all_id.append(tid)
            all_polarity.append(polarity)
            all_opinions.append(opinions)

        for curid in range(len(all_id)):
            term = term_to_position_list[all_id[curid]]['term']
            term_from = term_to_position_list[all_id[curid]]['from']
            term_to = term_to_position_list[all_id[curid]]['to']
            cur_polarity = all_polarity[curid]
            other_id = all_id[:curid] + all_id[curid + 1:]
            cur_opinions = term_to_position_list[all_id[curid]]['opinions']
            cur_opinions_positions = [i[1] for i in cur_opinions]

            change_words = words_list
            change_opi_words = []
            change_opinion_position = []

            for ix, id in enumerate(other_id):
                other_opinions = term_to_position_list[id]['opinions']
                other_opinions_positions = [i[1] for i in other_opinions]

                find = False
                for i in other_opinions_positions:
                    if i in cur_opinions_positions:
                        find = True
                        break
                if find:
                    continue

                non_overlap_opinions = []
                for op in other_opinions:
                    if op[1] in change_opinion_position:
                        continue
                    else:
                        non_overlap_opinions.append(op)

                if len(non_overlap_opinions) == 0:
                    continue

                if cur_polarity == term_list[other_id[ix]]['polarity']:
                    if term_list[other_id[ix]]['polarity'] == 'positive':
                        new_words, new_opi_words, _ = util.reverse(change_words,
                                                                   non_overlap_opinions, copy_sent)
                    elif term_list[other_id[ix]]['polarity'] == 'negative':
                        new_words, new_opi_words, _ = util.reverse(change_words,
                                                                   non_overlap_opinions, copy_sent)
                    else:
                        continue

                    and_ind = []
                    for i, w in enumerate(new_words):
                        if w.lower() in [
                            'and'] and w.lower() not in term.lower():
                            and_ind.append(i)

                    min = 1e8
                    aidx = -1
                    for idx in and_ind:
                        if idx > term_to and idx < len(new_words) - 3:
                            if idx - term_to < min:
                                min = idx - term_to
                                aidx = idx
                        if idx < term_from and idx > 3:
                            if term_to - idx < min:
                                min = term_to - idx
                                aidx = idx
                    if aidx != -1:
                        new_words[aidx] = 'but'
                else:
                    new_words, new_opi_words = util.exaggerate(change_words,
                                                               non_overlap_opinions)

                if new_words != change_words:
                    for op in new_opi_words:
                        change_opinion_position.append([op[0], op[1]])
                    change_words = new_words

            if change_words == words_list:
                continue

            new_sent = util.untokenize(change_words)

            a = ''.join(util.tokenize(new_sent))
            b = ''.join(util.tokenize(term))
            c = 0
            for i in range(len(a)):
                if a[i:i + len(b)] == b:
                    c = len(a[:i])
                    break

            span_from = 0
            c2 = 0
            for i in range(len(new_sent) - 1):
                if new_sent[i] != ' ':
                    c2 += 1
                if c2 == c and c != 0 and new_sent[i + 1] != ' ':
                    span_from = i + 1
                    break
                if c2 == c and c != 0 and new_sent[i + 1] == ' ':
                    span_from = i + 2
                    break

            span_to = span_from + len(term)

            if new_sent[span_from:span_to] != term:
                print(all_id[curid], term, new_sent[span_from:span_to])

            print(new_sent)

            if sentence[0].isupper():
                new_sent = new_sent[0].upper() + new_sent[1:]

            res[all_id[curid]] = {
                'term': term,
                'id': all_id[curid],
                'sentence': new_sent,
                'multi': sent_example['multi'],
                'contra': sent_example['contra'],
                'from': span_from,
                'to': span_to,
                'polarity': cur_polarity
            }

    with open(outfile, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)
    print(len(res))


# the aspectset is created from infile, and infile2 are the sentences to which the
def addDiff(dataset, infile, infile2, outfile, same=False):
    tool = language_tool_python.LanguageTool('en-US')
    util = Utils(dataset)
    with open(infile, 'r', encoding='utf-8') as fw:
        examples = json.load(fw)

    # aspectset(s)
    pos_noun_adj_pair = []
    neg_noun_adj_pair = []
    neu_noun_adj_pair = []

    ####################################### NEW IN ARTS++ ################################################

    for id in examples:
        example = examples[id]
        term_list = example['term_list']
        sentence = example['sentence']
        annotations = util.get_constituent(sentence)
        for i, tid in enumerate(term_list):
            term = term_list[tid]['term']
            if "category" in term_list[tid]:
                term_category = term_list[tid]['category']
            else:
                term_category = get_term_category(term, sentence)
            opinion = term_list[tid]['opinion_words'][-1].lower()
            try:
                # some sentence can not be parsed
                ptree = ParentedTree.fromstring(annotations)
            except:
                continue
            phrases = util.get_phrase(term, opinion, ptree)

            other_terms = []
            for otid in term_list:
                if otid != tid:
                    other_terms.append(
                        ''.join(term_list[otid]['term'].split(' ')))

            opi_words = []
            for p in phrases:
                p_ = ''.join(p)
                overlap = False
                for other_term in other_terms:
                    if other_term in p_:
                        overlap = True
                        break
                if not overlap:
                    opi_words.append(p)

            opi_words = sorted(opi_words, key=len)

            if len(opi_words) == 0:
                continue
            opi_words = opi_words[0]

            # ALSO CHANGES MADE
            if term_list[tid]['polarity'] == 'positive':
                pos_noun_adj_pair.append(
                    (term.lower(), [k.lower() for k in opi_words], term_category))
            elif term_list[tid]['polarity'] == 'negative':
                neg_noun_adj_pair.append(
                    (term.lower(), [k.lower() for k in opi_words], term_category))
            elif term_list[tid]['polarity'] == 'neutral':
                neu_noun_adj_pair.append(
                    (term.lower(), [k.lower() for k in opi_words], term_category))

    with open(infile2, 'r', encoding='utf-8') as fw:
        examples = json.load(fw)

    res = {}
    for id in examples:
        sent_example = examples[id]
        term_list = sent_example['term_list']
        all_term = []
        for tid in term_list:
            all_term.append(term_list[tid]['term'])

        for tid in term_list:
            sentence = sent_example['sentence']

            fromidx = term_list[tid]['from']
            toidx = term_list[tid]['to']

            term = term_list[tid]['term']
            target_term_category = get_term_category(term, sentence)
            polarity = term_list[tid]['polarity']
            if same:
                if polarity == 'positive':
                    pair = pos_noun_adj_pair
                elif polarity == 'negative':
                    pair = neg_noun_adj_pair
                else:
                    pair = neu_noun_adj_pair
            else:
                if polarity == 'positive':
                    pair = neg_noun_adj_pair
                elif polarity == 'negative':
                    pair = pos_noun_adj_pair
                else:
                    pair = neu_noun_adj_pair

            punct = '.'
            if sentence[-1] == string.punctuation:
                punct = sentence[-1]

            pair_current_category_list = [item for item in pair if item[2] == target_term_category]

            # we want each term to appear only once in the pair list
            seen_terms = set()
            pair_current_category = []

            for item in pair_current_category_list:
                cur_term = item[0]
                if cur_term not in seen_terms:
                    seen_terms.add(cur_term)
                    pair_current_category.append(item)

            max_len_adddiff = len(pair_current_category)

            # if there are no instances in the term's category with opposing sentiment
            # just perform the regular AddDiff steps
            if max_len_adddiff == 0:
                pair_current_category = pair
                max_len_adddiff = 3

            polarity_list = []
            repeated_counter = 0
            while True and repeated_counter < 50:
                repeated_counter += 1
                add_num = random.randint(min(1, max_len_adddiff), min(3, max_len_adddiff))
                if len(pair_current_category) == 1:
                    random_pair1 = pair_current_category[0]
                    random_pair2 = pair_current_category[0]
                    random_pair3 = pair_current_category[0]
                    add_num = 1
                elif len(pair_current_category) == 2:
                    random_pair1 = pair_current_category[0]
                    random_pair2 = pair_current_category[1]
                    random_pair3 = pair_current_category[1]
                    add_num = 2
                else:
                    random_num1, random_num2, random_num3 = random.sample(
                        range(len(pair_current_category)), 3)
                    random_pair1 = pair_current_category[random_num1]
                    random_pair2 = pair_current_category[random_num2]
                    random_pair3 = pair_current_category[random_num3]
                if (random_pair1[0] not in all_term and random_pair2[
                    0] not in all_term and random_pair3[
                        0] not in all_term) or repeated_counter == 50:
                    if random_pair1 in pos_noun_adj_pair:
                        polarity_list.append('positive')
                    elif random_pair1 in neg_noun_adj_pair:
                        polarity_list.append('negative')
                    else:
                        polarity_list.append('')

                    if random_pair2 in pos_noun_adj_pair:
                        polarity_list.append('positive')
                    elif random_pair2 in neg_noun_adj_pair:
                        polarity_list.append('negative')
                    else:
                        polarity_list.append('')

                    if random_pair3 in pos_noun_adj_pair:
                        polarity_list.append('positive')
                    elif random_pair3 in neg_noun_adj_pair:
                        polarity_list.append('negative')
                    else:
                        polarity_list.append('')
                    break

            polarity_dict = {'positive': 0, 'negative': 0}
            for tid_ in term_list:
                if term_list[tid_]['polarity'] in ['positive', 'negative']:
                    polarity_dict[term_list[tid_]['polarity']] += 1

            if add_num == 3:
                tmp_words = random_pair1[1] + [','] + random_pair2[1] + [
                    'and'] + random_pair3[1] + [punct]
                for m in polarity_list[:3]:
                    if len(m) > 0:
                        polarity_dict[m] += 1
            elif add_num == 2:
                tmp_words = random_pair1[1] + ['and'] + random_pair2[1] + [
                    punct]
                for m in polarity_list[:2]:
                    if len(m) > 0:
                        polarity_dict[m] += 1
            else:
                tmp_words = random_pair1[1] + [punct]
                for m in polarity_list[:1]:
                    if len(m) > 0:
                        polarity_dict[m] += 1

            while sentence[-1] in '.?!':
                sentence = sentence[:-1]

            opi_tag, opi_tag_uni = util.get_postag(tmp_words, 0, 1)

            if opi_tag_uni[0][1] != 'CONJ':
                tmp_sentence = 'but ' + util.untokenize(tmp_words)
                new_sentence = tool.correct(tmp_sentence)
                new_sentence = new_sentence[4:]

                if 'but' in sentence or 'although' in sentence:
                    new_sent = sentence + "; " + new_sentence
                else:
                    new_sent = sentence + ", but " + new_sentence
            else:
                tmp_sentence = util.untokenize(tmp_words)
                new_sentence = tool.correct(tmp_sentence)
                new_sent = sentence + ". " + new_sentence[
                    0].upper() + new_sentence[1:]

            # if new_sent[fromidx:toidx] != term:
            #     print("****not equal****")

            print(new_sent)
            if new_sent:
                res[tid] = {
                    'term': term,
                    'id': tid,
                    'sentence': new_sent,
                    'multi': sent_example['multi'],
                    'contra': sent_example['contra'],
                    'from': fromidx,
                    'to': fromidx + len(term),
                    'polarity': polarity,
                    'portion': polarity_dict
                }

    with open(outfile, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, indent=4)

