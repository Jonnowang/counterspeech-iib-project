import os
import json

__location__ = os.getcwd()
f = open(f"{__location__}/RUBER/data/gab_query.txt", "r")
queries = f.readlines()

words = 0
for query in queries:
    word_list = query.split()
    words += len(word_list)

print(words)
# for query in queries:
#     response = openai.Completion.create(
#         model="curie:ft-user-vh7kxsroswggzc53nnfafzav-2021-11-20-19-17-30",
#         prompt=f"Human: {query.strip()}\n AI:",
#         max_tokens=40,
#         temperature=0.9,
#         stop=["\n", "Human:", "AI:"],
#         presence_penalty=0.6)

#     reply = response["choices"][0]["text"]
#     print(reply)

#     with open(f"{__location__}/RUBER/data/gab_response_gpt3_tuned.txt", 'a') as fr:
#         fr.write(f"{reply}\n")