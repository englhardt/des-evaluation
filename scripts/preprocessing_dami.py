#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import logging
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.io.arff import loadarff

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

NOISE_AMOUNT = 0.01
RANDOM_STATE = 0
NUM_FEATURES = 5

input_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "input", "raw", "dami")
output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "input", "processed", "dami")
file_regex = re.compile('(.*withoutdupl_norm(_\d\d)?.arff$)')

def load_file(file_name):
    data, meta = loadarff(file_name)
    data = pd.DataFrame(data, columns=meta.names())
    data['label'] = data['outlier'].apply(lambda x: 'outlier' if x == b"'yes'" else 'inlier')
    data = data.drop(columns=['id', 'outlier'])
    # reorder columns
    data = data[np.append([x for x in data.columns if x != 'label'], ['label'])]
    return data

def feature_selection(data):
    # perform feature selection
    X = data[[x for x in data.columns if x != 'label']]
    y = data['label']
    feature_selection = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE), k=NUM_FEATURES)
    feature_selection.fit(X, y)
    column_mask = feature_selection.get_support()
    data_filtered = data.iloc[:, np.append(column_mask, [False])].reset_index(drop=True)
    data_filtered['label'] = y
    return data_filtered

def add_noise(data):
    # add small amount of noise
    np.random.seed(RANDOM_STATE)
    data += np.random.normal(scale=NOISE_AMOUNT, size=data.shape)
    return data

def process_file(input_file, output_file):
    data = load_file(input_file)
    data_filtered = feature_selection(data)
    data_filtered.iloc[:, :-1] = add_noise(data_filtered.iloc[:, :-1])
    data_filtered.to_csv(output_file, header=False, index=False)

def main():
    if not os.listdir(input_dir):
        logging.error(f"Data input dir {input_dir} is empty. Please follow the" + \
                       "instructions in the README to download the raw data.")
        return

    target_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file_regex.match(file):
                target_files += [os.path.join(root, file)]

    for f in target_files:
        logging.info(f"Processing '{f}.")
        output_file = f.replace(os.path.join(input_dir, 'semantic'), output_dir)
        output_file = f.replace(os.path.join(input_dir, 'literature'), output_dir)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        process_file(f, output_file)
    logging.info("Done.")

if __name__ == "__main__":
    main()
