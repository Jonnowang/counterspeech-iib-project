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

query = "i joined gab to remind myself how retarded jew haters are. You wouldn't be typing on your abacus without them you retard."

reference = ["Using words that insult one group while defending another group doesn't come across as helpful.", 
             "You can make the same point more effectively without the use of hateful terminology.", 
             "Use of the r-word is unacceptable in our discourse as it demeans and insults people with mental disabilities."]

GPT3_response = "I was wondering why you joined our site. As we're all individuals here and we live in a democracy, we're all free to express our ideas without fear of reprisal. Does that sound right?"

Blenderbot_response = "That's a good way to look at it. I don't think I'd be able to do that."

i = 0
query_file = list()
response_file = list()

for idx in gab_dataset["hate_speech_idx"].tolist():
    try:
        idx = ast.literal_eval(idx)
    except:
        idx = []
    try:
        resp = ast.literal_eval(gab_dataset["response"][i])
    except:
        resp = []

    text = gab_dataset["text"][i]
    filtered_list = list()
    for id in idx:
        r = re.compile(f"{id}\..*")
        hate_text = re.findall(r, text)
        filtered_list.append(hate_text)

    query_file.append(filtered_list)
    response_file.append(resp)
    i += 1

query_df = pd.DataFrame({"query": query_file})
response_df = pd.DataFrame({"response": response_file})

r = re.compile("\d\.")
r2 = re.compile(r"\n")

queries = query_df["query"].to_list()
responses = response_df["response"].to_list()

qr = []
for query, response in zip(queries[:1000], responses[:1000]):
    try:
        processed_query = re.sub(r, "", query[0][0])
    except IndexError:
        continue
    try:
        processed_response = re.sub(r2, "", response[1])
    except IndexError:
        continue
    qr.append({"prompt": processed_query.strip(), "completion": processed_response})
    with open(f"{__location__}/RUBER/data/gab_query_short_test.txt", 'a') as fq:
        fq.write(f"{processed_query.strip()}\n")
    with open(f"{__location__}/RUBER/data/gab_response_short_test.txt", 'a') as fr:
        fr.write(f"{processed_response}\n")

qr_json = json.dumps(qr)
with jsonlines.open(f"{__location__}/RUBER/data/gab_finetune_data.jsonl", 'a') as fjs:
    fjs.write_all(qr)

fq.close()
fr.close()
fjs.close()
