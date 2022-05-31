import os
from collections import defaultdict

import seaborn as sns
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords

__location__ = os.getcwd()

nltk.download('stopwords')
stop=set(stopwords.words('english'))
num_words=['1.', '2.', '3.', '4.', '5.']

gab = pd.read_csv('counterspeech_project-NLP/data/gab.csv')

with open(r"counterspeech_project-NLP\data\greedy_double_response.txt",'r') as f:
    responses = f.readlines()
# blender_responses = re.findall("   \[text\]: .*",blender_responses)
# for response in blender_responses:
#     filtered_response = re.sub("   \[text\]: ","",response)
#     with open(f"{__location__}/blender_responses.txt", 'a') as fw:
#         fw.write(f"{filtered_response}\n")

# gab_words = gab['text'].str.split().map(lambda x: len(x))
# print(gab_words.max())
# print(gab_words.min())
# gab_words.hist(bins=500, range=(0,1000))
# plt.show()

# reddit = pd.read_csv('counterspeech_project-NLP/data/reddit.csv')

# reddit_words = reddit['text'].str.split().map(lambda x: len(x))
# print(reddit_words.max())
# print(reddit_words.min())
# reddit_words.hist(bins=500, range=(0,1000))
# plt.show()

# conan = pd.read_json('counterspeech_project-NLP/data/CONAN.json')
# print(conan.head(3))

word_set = pd.DataFrame(responses)[0]
print(word_set)

corpus=[]
new= word_set.str.split()
new= new.values.tolist()
corpus=[word for i in new for word in i]

dic=defaultdict(int)
for word in corpus:
    if word in stop and word not in num_words:
        dic[word]+=1

print(len(dic))
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)
plt.bar(x,y)
plt.show()

# counter=Counter(corpus)
# most=counter.most_common()

# x, y= [], []
# for word,count in most[:10]:
#     if (word not in stop and word not in num_words):
#         x.append(word)
#         y.append(count)
        
# sns.barplot(x=y,y=x)
# plt.show()

def _get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    print(len(vec.vocabulary_.items()))
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                    for word, idx in vec.vocabulary_.items()]
    print(len(words_freq))
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq

top_n_bigrams=_get_top_ngram(word_set,2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
plt.show()