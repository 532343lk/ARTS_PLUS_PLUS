import numpy as np
import json
import xml.etree.ElementTree as ET
import codecs
import random
random.seed(1337)
np.random.seed(1337)

polar_idx={'positive': 0, 'negative': 1, 'neutral': 2}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}

def extract_spans(sentence, target_list):
    words = sentence.split()  # Split the sentence into words
    opinion_words = []
    opinion_position = []
    current_span = []
    span_start = None

    char_position = 0  # Initialize character position counter

    for idx, (word, label) in enumerate(zip(words, target_list)):
        if char_position > 0:  # Add space before word (except at the beginning)
            char_position += 1
        if label == 'B':
            if current_span:
                opinion_words.append(" ".join(current_span))
                opinion_position.append([span_start, char_position - 1])
                current_span = []
            current_span.append(word)
            span_start = char_position
        elif label == 'I':
            current_span.append(word)
        else:
            if current_span:
                opinion_words.append(" ".join(current_span))
                opinion_position.append([span_start, char_position - 1])
                current_span = []
                span_start = None
        if char_position > 0 and word[0] in "-:;.,!?()[]\'\"":
            char_position += len(word) - 1
        else:
            char_position += len(word)

    if current_span:
        opinion_words.append(" ".join(current_span))
        opinion_position.append([span_start, char_position])

    return opinion_words, opinion_position

def load_text_target_label(path):
    corpus = {}
    with codecs.open(path, encoding='utf-8') as fo:
        for i, line in enumerate(fo):
            if i == 0:
                continue
            s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
            w_t = target_tags.strip().split(' ')
            target = [t.split('\\')[-1] for t in w_t]
            w_l = opinion_words_tags.strip().split(' ')
            label = [l.split('\\')[-1] for l in w_l]
            target_word, target_position = extract_spans(sentence, target)
            opinion_words, opinion_position = extract_spans(sentence, label)
            corpus[s_id+target_word[0]] = {"opinion_words": opinion_words, "opinion_position": opinion_position}
    return corpus

def find_closest_occurrences(main_string, substrings, inaccurate_indices, id):
    occurrences = []
    accurate_indices = []
    
    for substring in substrings:
        substring_occurrences = []
        start_index = main_string.find(substring)
        
        while start_index != -1:
            end_index = start_index + len(substring) - 1
            substring_occurrences.append((start_index, end_index))
            start_index = main_string.find(substring, start_index + 1)
        
        occurrences.append(substring_occurrences)
    
    closest_occurrences = []
    
    for i, substring_occurrences in enumerate(occurrences):
        closest_occurrence = None
        min_distance = float('inf')
        
        for start, end in substring_occurrences:
            inaccurate_start_index = inaccurate_indices[i][0]
            distance = abs(inaccurate_start_index - start)
            if distance < min_distance:
                min_distance = distance
                closest_occurrence = (start, end)
        
        closest_occurrences.append(closest_occurrence)
    
    for i, closest_occurrence in enumerate(closest_occurrences):
        substring = substrings[i]
        if closest_occurrence:
            start_index, end_index = closest_occurrence
            accurate_indices.append([start_index, end_index+1])
        else:
            accurate_indices.append(inaccurate_indices[i])
            print(f"Occurrence of substring '{substring}' not found in: {id}")

    return accurate_indices

def parse_SemEval(version, fn, corpus_opinions):

    if version ==  14:
        SemEval = ['aspectTerm','term',]
    elif version == 15:
        SemEval = ['Opinion','target']
    else:
        print("Version not found ...")
        return
    
    root=ET.parse(fn).getroot()
    # print(corpus_opinions)
    corpus=[]
    opin_cnt=[0]*len(polar_idx)
    for sent in root.iter("sentence"):
        sent_opin_cnt=[0]*len(polar_idx)
        multi = True
        contra = True
        term_list={}
        opins=set()
        for opin in sent.iter(SemEval[0]):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib[SemEval[1]]!="NULL":
                if opin.attrib['polarity'] in polar_idx:
                    opins.add((opin.attrib[SemEval[1]], int(opin.attrib['from']), int(opin.attrib['to']), opin.attrib['polarity'] ) )  
                    sent_opin_cnt[polar_idx[opin.attrib['polarity']]] += 1
        if len(opins) == 1:
            multi = False
        if (sent_opin_cnt[0] == len(opins)) or (sent_opin_cnt[1] == len(opins)) or (sent_opin_cnt[2] == len(opins)):
            contra = False
        opins_list = list(opins)
        opins_list.sort(key=lambda x: x[3])
        opins_list.sort(key=lambda x: x[1])
        for ix, opin in enumerate(opins_list):
            opin_cnt[polar_idx[opin[3]]] += 1
            if sent.attrib['id']+opin[0] in corpus_opinions:

                opinion_words = corpus_opinions[sent.attrib['id']+opin[0]]["opinion_words"]
                modified_opinion_words = []

                for opinion_word in opinion_words:
                    modified_opinion_word = opinion_word.replace("can not", "cannot").replace(" n't", "n't")
                    modified_opinion_words.append(modified_opinion_word)

                inaccurate_indices = corpus_opinions[sent.attrib['id']+opin[0]]["opinion_position"]
                main_string = sent.find('text').text

                accurate_indices = find_closest_occurrences(main_string, modified_opinion_words, inaccurate_indices, sent.attrib['id'])
                term_list[sent.attrib['id']+"_"+str(ix)] = {"id": sent.attrib['id']+"_"+str(ix), "polarity": opin[-1], "term": opin[0], "from": opin[1], "to": opin[2], "opinion_words": modified_opinion_words, "opinion_position": accurate_indices}
            # else:
            #     term_list[sent.attrib['id']+"_"+str(ix)] = {"id": sent.attrib['id']+"_"+str(ix), "polarity": opin[-1], "term": opin[0], "from": opin[1], "to": opin[2]}
        if bool(term_list):
            corpus.append({"sentence": sent.find('text').text, "term_list": term_list, "contra": contra, "multi": multi, "id": sent.attrib['id']})
    # print(opin_cnt)
    # print(len(corpus))
    return corpus

def convert_xml_to_jason_sent_towe(version, dataset):

    train_corpus=parse_SemEval(version, "data/semval/{}/train.xml" .format(dataset), load_text_target_label("data/towe/{}/train.tsv" .format(dataset)))
    with open("data/generated/{}/train_sent_towe.json" .format(dataset), "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus }, fw, sort_keys=True, indent=4)
    test_corpus=parse_SemEval(version, "data/semval/{}/test.xml" .format(dataset), load_text_target_label("data/towe/{}/test.tsv" .format(dataset)))
    with open("data/generated/{}/test_sent_towe.json" .format(dataset), "w") as fw:
        json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
    
    return

convert_xml_to_jason_sent_towe(14, "14lap")
convert_xml_to_jason_sent_towe(14, "14res")
convert_xml_to_jason_sent_towe(15, "15res")
convert_xml_to_jason_sent_towe(15, "16res")