import ast
import json
import os
from collections import Counter

import jsonlines
import numpy as np
import pandas as pd
import regex as re

__location__ = os.getcwd()
reddit_dataset = pd.read_csv(f"{__location__}/counterspeech_project-NLP/data/reddit.csv")
gab_dataset = pd.read_csv(f"{__location__}/counterspeech_project-NLP/data/gab.csv")

existing = dict()

for responses in gab_dataset['response']:
    try:
        responses = ast.literal_eval(responses)
    except:
        responses = []
    for response in responses:
        try: 
            if existing[response] == 1:
                pass
        except KeyError:
            existing[response] = 1
            with open(f"{__location__}/counter_speech_cand.txt", 'a') as f:
                f.write(f"{response.strip()}\n")

for responses in reddit_dataset['response']:
    try:
        responses = ast.literal_eval(responses)
    except:
        responses = []
    for response in responses:
        try: 
            if existing[response] == 1:
                pass
        except KeyError:
            existing[response] = 1
            with open(f"{__location__}/counter_speech_cand.txt", 'a') as f:
                f.write(f"{response.strip()}\n")

# i = 0
# query_file = list()
# response_file = list()

# for idx in gab_dataset["hate_speech_idx"].tolist():
#     try:
#         idx = ast.literal_eval(idx)
#     except:
#         idx = []
#     try:
#         resp = ast.literal_eval(gab_dataset["response"][i])
#     except:
#         resp = []

#     text = gab_dataset["text"][i]
#     filtered_list = list()
#     for id in idx:
#         r = re.compile(f"{id}\..*")
#         hate_text = re.findall(r, text)
#         filtered_list.append(hate_text)

#     query_file.append(filtered_list)
#     response_file.append(resp)
#     i += 1

# query_df = pd.DataFrame({"query": query_file})
# response_df = pd.DataFrame({"response": response_file})

# r = re.compile("\d\.")
# r2 = re.compile(r"\n")

# queries = query_df["query"].to_list()
# responses = response_df["response"].to_list()

# qr = []
# for query, response in zip(queries[:1000], responses[:1000]):
#     try:
#         processed_query = re.sub(r, "", query[0][0])
#     except IndexError:
#         continue
#     try:
#         processed_response = re.sub(r2, "", response[1])
#     except IndexError:
#         continue
#     qr.append({"prompt": processed_query.strip(), "completion": processed_response})
#     with open(f"{__location__}/RUBER/data/gab_query_short_test.txt", 'a') as fq:
#         fq.write(f"{processed_query.strip()}\n")
#     with open(f"{__location__}/RUBER/data/gab_response_short_test.txt", 'a') as fr:
#         fr.write(f"{processed_response}\n")

# qr_json = json.dumps(qr)
# with jsonlines.open(f"{__location__}/RUBER/data/gab_finetune_data.jsonl", 'a') as fjs:
#     fjs.write_all(qr)

# fq.close()
# fr.close()
# fjs.close()
