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
        opins=set()
        for opin in sent.iter(SemEval[0]):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib[SemEval[1]]!="NULL":
                if opin.attrib['polarity'] in polar_idx:
                    opins.add((opin.attrib[SemEval[1]], int(opin.attrib['from']), int(opin.attrib['to']), opin.attrib['polarity'] ) )  
                    sent_opin_cnt[polar_idx[opin.attrib['polarity']]] += 1
        opins_list = list(opins)
        opins_list.sort(key=lambda x: x[3])
        opins_list.sort(key=lambda x: x[1])
        for ix, opin in enumerate(opins_list):
            opin_cnt[polar_idx[opin[3]]] += 1
            corpus.append({"id": sent.attrib['id']+"_"+str(ix), "sentence": sent.find('text').text, "term": opin[0], "from": opin[1], "to": opin[2], "polarity": opin[3]})
    return corpus

def convert_xml_to_jason(version, dataset):

    train_corpus=parse_SemEval(version, "data/semval/{}/train.xml" .format(dataset))
    with open("data/generated/{}/train.json" .format(dataset), "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus }, fw, sort_keys=True, indent=4)
    test_corpus=parse_SemEval(version, "data/semval/{}/test.xml" .format(dataset))
    with open("data/generated/{}/test.json" .format(dataset), "w") as fw:
        json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
    
    return

convert_xml_to_jason(14, "14lap")
convert_xml_to_jason(14, "14res")
convert_xml_to_jason(15, "15res")
convert_xml_to_jason(15, "16res")