import json
import pandas as pd
import nltk
import re


def combine_truth_files(ontology_path, neural_path, output_file_path):
    with open(ontology_path, 'r') as ont_file, open(neural_path, 'r') as neural_file, open(output_file_path,
                                                                                           'w') as output_file:
        for line1, line2 in zip(ont_file, neural_file):
            line1 = line1.strip()
            line2 = line2.strip()

            if line1 == "False":
                output_file.write("False\n")
            elif line1 == "True" or line2 == "True":
                output_file.write("True\n")
            else:
                output_file.write("False\n")
        output_file.close()


def calculate_ARS(input_file, result_file):
    year = 2014

    with open(input_file, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    df = pd.DataFrame(data).transpose()

    df['sid'] = df['id'].str.replace(r'_adv[1-3]', '', regex=True)
    df['adv'] = df['id'].str.extract(r'_adv([1-3])').fillna(0).astype(int)
    df.drop(['id', 'from', 'to', 'term', 'sentence'], axis=1, inplace=True)

    df['c'] = pd.read_csv(result_file, sep=" ", header=None).set_index(df.index)
    df['nc'] = ~df['c']

    acc = sum(df['c']) / len(df['c'])
    df_ARS = pd.DataFrame(df[['sid', 'nc']].groupby('sid').sum())
    ARS = len(df_ARS.loc[df_ARS['nc'] == 0]) / len(df_ARS['nc'])

    df = df[df['adv'] == 0]
    acc_0 = sum(df['c']) / len(df['c'])

    res = pd.DataFrame(
        {
            'acc': [acc],
            'acc_0': [acc_0],
            'ARS': [ARS]
        },
        index=['{}_{}'.format(year, set)])
    return res


def calculate_accuracy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    true_count = 0
    total_count = 0
    unsure_count = 0

    for line in lines:
        if line == "True\n":
            true_count += 1
            total_count += 1
        if line == "False\n":
            total_count += 1
        if line == "Unsure\n":
            unsure_count += 1
    print(true_count / total_count)
    print(unsure_count)


def loadFile(file, data):
    with open(file, 'r', encoding='utf-8') as fr:
        lines = json.load(fr)
        for sid in lines:
            line = lines[sid]

            term_from = line['from']
            term_to = line['to']
            sentence = line['sentence']
            sentenceNew = sentence[:term_from] + "$t$" + sentence[term_to:]
            sentenceNew2 = re.sub(' +', ' ', sentenceNew)
            sptoks = nltk.word_tokenize(sentenceNew2)
            outputt = ' '.join(sp for sp in sptoks).lower()
            outputtext = re.sub('\$ t \$', '$T$', outputt)
            data.append(outputtext)

            asp = line['term']
            aspNew = re.sub(' +', ' ', asp)
            t_sptoks = nltk.word_tokenize(aspNew)
            outputtarget = ' '.join(sp for sp in t_sptoks).lower()
            data.append(outputtarget)

            polarity = line['polarity']
            if polarity == 'negative':
                lab = -1
            elif polarity == 'neutral':
                lab = 0
            elif polarity == "positive":
                lab = 1
            data.append(str(lab))
    return data


def loadFiles(train, test):
    data = []
    data = loadFile(train, data)
    train_length = len(data)
    data = loadFile(test, data)

    with open('data/HAABSA_input_files/LaptopARTS++complete.txt', 'w') as file:
        for line in data:
            file.write(line + '\n')
    with open('data/HAABSA_input_files/LaptopARTS++train.txt', 'w') as trainfile:
        for i, line in enumerate(data):
            if i < train_length:
                trainfile.write(line + '\n')
    with open('data/HAABSA_input_files/LaptopARTS++test.txt', 'w') as testfile:
        for i, line in enumerate(data):
            if i >= train_length:
                testfile.write(line + '\n')

    return data

