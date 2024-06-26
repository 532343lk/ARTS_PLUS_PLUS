# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
from strategies import revTgt, revNon, addDiff

if __name__ == '__main__':

    input_aspectset = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop/test_sent_towe.json"
    input_file = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop/test_sent.json"
    output_file_revtgt = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop/output/ARTS++/test/revTgt.json"
    output_file_revnon = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop/output/ARTS++/test/revnon.json"
    output_file_adddiff = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop/output/ARTS++/test/adddiff.json"
    data_folder = "/Users/lorenzkremer/Documents/MasterThesis/data/2014/laptop/"
    dataset = ""
    strategy = "all"


    if strategy == 'all':
        revTgt(data_folder, input_aspectset, output_file_revtgt)
        revNon(data_folder, input_aspectset, output_file_revnon)
        addDiff(data_folder, input_aspectset, input_file, output_file_adddiff)
    if strategy == 'revTgt':
        revTgt(data_folder, input_file, output_file_revtgt)
    elif strategy == 'revNon':
        revNon(data_folder, input_file, output_file_revnon)
    elif strategy == 'addDiff':
        addDiff(data_folder, input_aspectset,
                input_file, output_file_adddiff)
