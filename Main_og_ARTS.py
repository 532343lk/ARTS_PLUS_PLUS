# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
from strategies_ARTS import revTgt, revNon, addDiff

if __name__ == '__main__':

    input_aspectset = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/rest/test_sent_towe.json"
    input_file = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/rest/test_sent.json"
    output_file = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/rest/output_ARTS/test/addDiff.json"
    data_folder = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/rest/"
    dataset = ""
    strategy = "addDiff"

    if strategy == 'revTgt':
        revTgt(data_folder, input_file, output_file)
    elif strategy == 'revNon':
        revNon(data_folder, input_file, output_file)
    elif strategy == 'addDiff':
        addDiff(data_folder, input_aspectset,
                input_file, output_file)
