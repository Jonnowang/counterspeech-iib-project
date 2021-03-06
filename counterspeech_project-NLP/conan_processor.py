import json
import os

import pandas as pd
import regex as re

__location__ = os.getcwd()

with open(f"{__location__}/counterspeech_project-NLP/data/CONAN.json",'r') as f:
    conan_data = json.loads(f.read())
conan_df = pd.json_normalize(conan_data, record_path =['conan'])

counters = []
for lang, hate, counter, typecn in zip(conan_df["cn_id"], conan_df['hateSpeech'], conan_df['counterSpeech'], conan_df['cnType']):
    if lang[:2] != 'EN': continue
    # if typecn == 'humor':
    if counter not in counters:
        counters.append(counter)
        with open(f"{__location__}/conan_data_train.txt", 'a') as fw:
            fw.write(f"text:{hate.strip()}\tlabels:{counter.strip()}\tepisode_done:{True}\n")
        with open(f"{__location__}/conan_data_valid.txt", 'a') as fw2:
            fw2.write(f"text:{hate.strip()}\tlabels:{counter.strip()}\tepisode_done:{True}\n")