import os
import json
from strategies_new import revTgt,revNon,addDiff

def make_adv(dataset, split):

    data_folder = 'data/generated/{}/'.format(dataset)

    input_file = os.path.join(data_folder, '{}_sent_towe.json' .format(split))
    infile = os.path.join(data_folder, 'train_sent_towe.json')
    infile2 = os.path.join(data_folder, '{}_sent.json' .format(split))
    output_file = os.path.join(data_folder, '{}_adv.json' .format(split))
    res_file = os.path.join(data_folder, '{}.json' .format(split))
        
    res1 = revTgt({}, data_folder, input_file)
    with open(os.path.join(data_folder, '{}_adv1.json' .format(split)), 'w', encoding='utf-8') as fw:
            json.dump(res1, fw, sort_keys=True, indent=4)
    res2 = revNon({}, data_folder, input_file)
    with open(os.path.join(data_folder, '{}_adv2.json' .format(split)), 'w', encoding='utf-8') as fw:
        json.dump(res2, fw, sort_keys=True, indent=4)
    res3 = addDiff({}, data_folder, infile, infile2)
    with open(os.path.join(data_folder, '{}_adv3.json' .format(split)), 'w', encoding='utf-8') as fw:
        json.dump(res3, fw, sort_keys=True, indent=4)

    # with open(os.path.join(data_folder, '{}_adv1.json' .format(split)), 'r', encoding='utf-8') as fr:
    #     res1 = json.load(fr)
    # with open(os.path.join(data_folder, '{}_adv2.json' .format(split)), 'r', encoding='utf-8') as fr:
    #     res2 = json.load(fr)
    # with open(os.path.join(data_folder, '{}_adv3.json' .format(split)), 'r', encoding='utf-8') as fr:
    #     res3 = json.load(fr)

    with open(res_file, 'r', encoding='utf-8') as fr:
        res4 = json.load(fr)
    
    res = {**res1, **res2, **res3, **res4}
    
    with open(output_file, 'w', encoding='utf-8') as fw:
        json.dump(res, fw, sort_keys=True, indent=4)

make_adv("14lap", "train")
make_adv("14lap", "test")

make_adv("14res", "train")
make_adv("14res", "test")

make_adv("15res", "train")
make_adv("15res", "test")

make_adv("16res", "train")
make_adv("16res", "test")