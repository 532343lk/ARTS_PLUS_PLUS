import numpy as np


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
