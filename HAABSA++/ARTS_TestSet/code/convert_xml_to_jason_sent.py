import numpy as np
import json
import xml.etree.ElementTree as ET
import random
random.seed(1337)
np.random.seed(1337)

polar_idx={'positive': 0, 'negative': 1, 'neutral': 2}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}

def parse_SemEval(version, fn):

    if version ==  14:
        SemEval = ['aspectTerm','term',]
    elif version == 15:
        SemEval = ['Opinion','target']
    else:
        print("Version not found ...")
        return
    
    root=ET.parse(fn).getroot()
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
            term_list[sent.attrib['id']+"_"+str(ix)] = {"id": sent.attrib['id']+"_"+str(ix), "polarity": opin[-1], "term": opin[0], "from": opin[1], "to": opin[2]}
        if bool(term_list):
            corpus.append({"sentence": sent.find('text').text, "term_list": term_list, "multi": multi, "contra": contra, "id": sent.attrib['id']})
    # print(opin_cnt)
    # print(len(corpus))
    return corpus

def convert_xml_to_jason_sent(version, dataset):

    train_corpus=parse_SemEval(version, "data/semval/{}/train.xml" .format(dataset))
    with open("data/generated/{}/train_sent.json" .format(dataset), "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus }, fw, sort_keys=True, indent=4)
    test_corpus=parse_SemEval(version, "data/semval/{}/test.xml" .format(dataset))
    with open("data/generated/{}/test_sent.json" .format(dataset), "w") as fw:
        json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)

    return

convert_xml_to_jason_sent(14, "14lap")
convert_xml_to_jason_sent(14, "14res")
convert_xml_to_jason_sent(15, "15res")
convert_xml_to_jason_sent(15, "16res")