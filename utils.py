# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
import re
import json
import nltk
import random
import stanza
from copy import deepcopy
from allennlp.predictors.predictor import Predictor
from pywsd import simple_lesk
import numpy as np

stanza.download('en')


class Utils():
    def __init__(self, data_folder='/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop'):
        self.negative_words_list = [
            'doesn\'t', 'don\'t', 'didn\'t', 'no', 'did not', 'do not',
            'does not', 'not yet', 'not', 'none', 'no one', 'nobody', 'nothing',
            'neither', 'nowhere', 'never', 'hardly', 'scarcely', 'barely'
        ]
        self.negative_words_list = sorted(self.negative_words_list,
                                          key=lambda s: len(s), reverse=True)
        self.degree_word_list = [
            'absolutely', 'awfully', 'badly', 'barely', 'completely',
            'decidedly', 'deeply', 'enormously', 'entirely', 'extremely',
            'fairly', 'fully',
            'greatly', 'highly',
            'incredibly', 'indeed', 'very', 'really'
        ]
        text = self.read_text(
            [os.path.join(data_folder, 'train_sent.json'),
             os.path.join(data_folder, 'dev_sent.json'),
             os.path.join(data_folder, 'test_sent.json'), ])

        self.word2idx = self.get_word2id(text)
        self.predictor = Predictor.from_path(
            "/Users/lorenzkremer/Documents/MasterThesis/elmo-constituency-parser-2020.02.10")
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos',
                                   tokenize_pretokenized=True)

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def untokenize(self, words):
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = ' '.join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',
                                                                     '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ").replace(' - ',
                                                                        '-').replace(
            ' / ', '/')
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
            "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        step7 = step6.replace("DELETE", "")
        step8 = re.sub(r"\s{2,}", " ", step7)
        return step8.strip()

    def tokenize_term_list(self, copy_sent, sent_example):
        term_to_position_list = {}
        term_list = sent_example['term_list']  # list of terms (aspects) in the current sentence
        for tid in term_list:
            if tid not in term_to_position_list:
                term_to_position_list[tid] = {}
            opinion_to_position_list = []
            opinions = term_list[tid]['opinion_words']
            opinions_spans = term_list[tid]['opinion_position']
            polarity = term_list[tid]['polarity']
            for i in range(len(opinions)):
                posi = opinions_spans[i]
                opi_from = posi[0]
                opi_to = posi[1]
                left = self.tokenize(copy_sent[:opi_from].strip())
                opi = self.tokenize(copy_sent[opi_from:opi_to].strip())
                opinion_to_position_list.append(
                    [' '.join(opi), [len(left), len(left) + len(opi)]])

            fromidx = term_list[tid]['from']
            toidx = term_list[tid]['to']
            left = self.tokenize(copy_sent[:fromidx].strip())
            aspect = self.tokenize(copy_sent[fromidx:toidx].strip())
            term_to_position_list[tid]['id'] = tid
            term_to_position_list[tid]['term'] = term_list[tid]['term']
            term_to_position_list[tid]['from'] = len(left)
            term_to_position_list[tid]['to'] = len(left) + len(aspect)
            term_to_position_list[tid]['polarity'] = polarity
            term_to_position_list[tid]['opinions'] = opinion_to_position_list
        return term_to_position_list

    def reverse(self, words_list, opinions, sentence):
        added_not = False
        new_words = deepcopy(words_list)
        new_opi_words = []
        from_to = []
        for i in range(len(opinions)):
            opi = opinions[i][0]
            opi_position = opinions[i][1]
            opi_from = opi_position[0]
            opi_to = opi_position[1]

            has_neg = False

            for w in self.negative_words_list:
                ws = self.tokenize(w)
                for j in range(opi_from, opi_to - len(ws) + 1):
                    new_words_ = ' '.join(new_words[j:j + len(ws)])
                    ws_ = ' '.join(ws)
                    if new_words_.lower() == ws_.lower():
                        if j > opi_from:
                            opi_to = opi_to - len(ws)
                            new_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                            has_neg = True
                            break
                        else:
                            opi_from = j + len(ws)
                            new_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                            has_neg = True
                            break
                if has_neg:
                    break
            opi_list = new_words[opi_from:opi_to]
            opi_tag, opi_tag_uni = self.get_postag(new_words, opi_from, opi_to)

            pos_translator_dict = {"ADJ": "a", "ADV": "r", "NOUN": "n", "VERB": "v"}

            opi_words = new_words[opi_from:opi_to]
            if len(opi_list) == 1:
                opi = opi_list[0]

                translated_pos_tag = None
                if opi_tag_uni[0][-1] in pos_translator_dict:
                    translated_pos_tag = pos_translator_dict[opi_tag_uni[0][-1]]

                if has_neg:
                    # delete negation words
                    if [opi_from, opi_to] not in from_to:
                        new_opi_words.append(
                            [opi_from, opi_to, self.untokenize(opi_words)])
                        from_to.append([opi_from, opi_to])
                else:
                    candidate = self.get_antonym_words(opi, sentence, translated_pos_tag)
                    refined_candidate = self.refine_candidate(new_words,
                                                              opi_from, opi_to,
                                                              candidate)

                    if len(refined_candidate) == 0:
                        # negate the closest verb
                        opi_tag2, opi_tag_uni2 = self.get_postag(new_words, 0,
                                                                 -1)
                        if opi_tag_uni[0][-1] == 'ADJ' or opi_tag_uni[0][
                            -1] == 'NOUN' or opi_tag_uni[0][-1] == 'VERB':
                            if [opi_from, opi_to] not in from_to:
                                new_opi_words.append([opi_from, opi_to,
                                                      self.untokenize(
                                                          ['not', opi])])
                                from_to.append([opi_from, opi_to])
                        else:
                            dis = 1e10
                            fidx = -1
                            for idx, (w, t) in enumerate(opi_tag2):
                                if abs(idx - opi_from) < dis and w in ['is',
                                                                       'was',
                                                                       'are',
                                                                       'were',
                                                                       'am',
                                                                       'being']:
                                    dis = abs(idx - opi_from)
                                    fidx = idx
                            if fidx == -1:
                                if [opi_from, opi_to] not in from_to:
                                    new_opi_words.append([opi_from, opi_to,
                                                          self.untokenize(
                                                              ['not', opi])])
                                    from_to.append([opi_from, opi_to])
                            else:
                                if [opi_from, opi_to] not in from_to:
                                    new_opi_words.append(
                                        [fidx, fidx + 1, self.untokenize(
                                            [opi_tag_uni2[fidx][0], 'not'])])
                                    from_to.append([opi_from, opi_to])
                        added_not = True
                    else:
                        select = random.randint(0, len(refined_candidate) - 1)
                        if [opi_from, opi_to] not in from_to:
                            new_opi_words.append([opi_from, opi_to,
                                                  self.untokenize([
                                                      refined_candidate[
                                                          select]])])
                            from_to.append([opi_from, opi_to])
            elif len(opi_list) > 1:
                if has_neg:
                    new_opi_words.append(
                        [opi_from, opi_to, self.untokenize(opi_words)])
                else:
                    # negate the closest verb
                    new_opi_words.append(
                        [opi_from, opi_to, self.untokenize(
                            ['not ' + opi_words[0]] + opi_words[1:])])
                    added_not = True

        for nopi in new_opi_words:
            new_words[nopi[0]:nopi[1]] = [nopi[2]]
        return new_words, new_opi_words, added_not

    def exaggerate(self, words_list, opinions):
        new_words = deepcopy(words_list)
        new_opi_words = []
        for i in range(len(opinions)):
            opi_position = opinions[i][1]
            opi_from = opi_position[0]
            opi_to = opi_position[1]

            new_words = self.add_degree_words(new_words, opi_from, opi_to)
            new_opi_word = self.untokenize(new_words[opi_from:opi_to])
            new_opi_words.append([opi_from, opi_to, new_opi_word])

        return new_words, new_opi_words

    # gives 2 types of POS tags. Simple is the one we're looking for most of the time
    def get_postag(self, x, s, e):
        # TODO
        doc = self.nlp([x])
        tags = [word.xpos for sent in doc.sentences for word in sent.words]
        simple_tags = [word.upos for sent in doc.sentences for word in
                       sent.words]
        words = [word.text for sent in doc.sentences for word in sent.words]

        t1 = list(zip(words, tags))
        t2 = list(zip(words, simple_tags))
        if e != -1:
            return t1[s:e], t2[s:e]
        else:
            return t1[s:], t2[s:]

    def get_antonym_words(self, word, sentence, pos):
        antonyms = set()
        found_antonyms = False
        synset = simple_lesk(sentence, word, pos=pos)
        if synset:
            for lemma in synset.lemmas():
                for a in lemma.antonyms():
                    antonyms.add(a.name())
                    found_antonyms = True
            if not found_antonyms:
                for s in synset.similar_tos():
                    for lemma in s.lemmas():
                        for a in lemma.antonyms():
                            antonyms.add(a.name())
        return antonyms

    def refine_candidate(self, words_list, opi_from, opi_to, candidate_list):
        if len(words_list) == 0:
            return []
        postag_list, _ = self.get_postag(words_list, 0, -1)
        postag_list = [t[1] for t in postag_list]
        in_vocab_candidate_list = []
        for candidate in candidate_list:
            if candidate.lower() in self.word2idx:
                in_vocab_candidate_list.append(candidate)

        refined_candi = []
        for candidate in in_vocab_candidate_list:
            opi = words_list[opi_from:opi_to][0]
            isupper = opi[0].isupper()
            allupper = opi.isupper()

            if allupper:
                candidate = candidate.upper()
            elif isupper:
                candidate = candidate[0].upper() + candidate[1:]
            if opi_from == 0:
                candidate = candidate[0].upper() + candidate[1:]

            new_words = words_list[:opi_from] + [candidate] + words_list[
                                                              opi_to:]

            # check pos tag
            new_postag_list, _ = self.get_postag(new_words, 0, -1)
            new_postag_list = [t[1] for t in new_postag_list]

            if len([i for i, j in zip(postag_list[opi_from:opi_to],
                                      new_postag_list[opi_from:opi_to]) if
                    i != j]) != 0:
                continue

            refined_candi.append(candidate)

        if len(refined_candi) == 0:
            for candidate in candidate_list:
                opi = words_list[opi_from:opi_to][0]
                isupper = opi[0].isupper()
                allupper = opi.isupper()

                if allupper:
                    candidate = candidate.upper()
                elif isupper:
                    candidate = candidate[0].upper() + candidate[1:]
                if opi_from == 0:
                    candidate = candidate[0].upper() + candidate[1:]

                new_words = words_list[:opi_from] + [candidate] + words_list[
                                                                  opi_to:]

                # check pos tag
                new_postag_list, _ = self.get_postag(new_words, 0, -1)
                new_postag_list = [t[1] for t in new_postag_list]

                if len([i for i, j in zip(postag_list[opi_from:opi_to],
                                          new_postag_list[opi_from:opi_to]) if
                        i != j]) != 0:
                    continue

                refined_candi.append(candidate)
        return refined_candi

    def get_word2id(self, text, lower=True):
        word2idx = {}
        idx = 1
        if lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
        return word2idx

    def read_text(self, fnames):
        text = ''
        for fname in fnames:
            with open(fname, 'r') as f:
                lines = json.load(f)
            for id in lines:
                instance = lines[id]
                text_instance = instance['sentence']
                # print(text_instance)
                text_raw = " ".join(self.process_text(text_instance)).lower()
                text += text_raw + " "
        return text.strip()

    def process_text(self, x):
        x = x.lower()
        x = x.replace("&quot;", " ")
        x = x.replace('"', " ")
        x = re.sub('[^A-Za-z0-9]+', ' ', x)
        x = x.strip().split(' ')
        # x = [strip_punctuation(y) for y in x]
        ans = []
        for y in x:
            if len(y) == 0:
                continue
            ans.append(y)
        # ptxt = nltk.word_tokenize(ptxt)
        return ans

    def add_degree_words(self, word_list, from_idx, to_idx):
        candidate_list = self.degree_word_list
        select = random.randint(0, len(candidate_list) - 1)
        opi = [' '.join([candidate_list[select]] + word_list[from_idx:to_idx])]
        new_words = word_list[:from_idx] + opi + word_list[to_idx:]
        return new_words

    def get_constituent(self, x):
        annotations = self.predictor.predict(sentence=x)['trees']
        return annotations

    def get_phrase(self, word, opi, ptree):
        phrase_level = [
            'ASJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX',
            'PP', 'PRN', 'PRT', 'QP', 'RRC',
            'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'S', 'SBAR'
        ]
        phrase = []
        for node in ptree.subtrees(filter=lambda t: t.label() in phrase_level):
            if node.label() == 'NP':
                if node.right_sibling() != None and node.right_sibling().label() == 'VP':
                    continue
            if node.label() == 'VP':
                if node.left_sibling() != None and node.left_sibling().label() == 'NP':
                    continue
            if ''.join(word.split(' ')) in ''.join(node.leaves()) and ''.join(
                    opi.split(' ')) in ''.join(node.leaves()):
                phrase.append(node.leaves())
        phrase = sorted(phrase, key=len, reverse=True)
        return phrase



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
    if type_ == 'TC':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), \
            np.asarray(x_r, dtype="object"), np.asarray(sen_len_r, dtype="object"), \
            np.asarray(y, dtype="object"), np.asarray(target_words, dtype="object"), \
            np.asarray(tar_len, dtype="object"), np.asarray(all_sent, dtype="object"), \
            np.asarray(all_target, dtype="object"), np.asarray(all_y, dtype="object")


def load_inputs_twitter_test(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10,
                             encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file

    x, sen_len = [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent = [], []
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
    if type_ == 'TC':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), \
            np.asarray(x_r, dtype="object"), np.asarray(sen_len_r, dtype="object"), \
            np.asarray(target_words, dtype="object"), np.asarray(tar_len, dtype="object"), \
            np.asarray(all_sent, dtype="object"), np.asarray(all_target, dtype="object")
