import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import re

df = pd.read_csv("counterspeech_project-NLP\Counter Speech Selection Survey.csv")
sers = df.sum(axis=0)
human_scores = np.array([i for i in sers[1:]])

min_max = MinMaxScaler()
human_scaled = min_max.fit_transform(human_scores.reshape(-1,1))
# print(human_scaled)

# from bert_score import score, plot_example

# cands = ['We are all global citizens, and we should be mindful of that in the 21st century.']
# refs = ['stop calling women names.']

# P, R, F1 = score(cands, refs, lang='en', verbose=True)
# print(F1)
# plot_example(cands[0], refs[0], lang="en")

# resp = df.columns

# for col in resp:
#     col = re.match("Hate Speech:.*\/",col)
#     if col is not None:
#         col = col.group(0)
#         print(f"{col[13:-9]}\n")

w2v = np.array([0.9999,  
0.8657,  
0.7876,  
0.7774,  
0.9999,  
0.8718,  
0.8692,  
0.9999,  
0.7818,  
0.8386, 
1.0000, 
0.6964, 
1.0000, 
0.0000, 
0.8168, 
0.9999, 
0.7711, 
0.3583, 
0.8671, 
0.8398, 
0.8628, 
0.8784, 
0.8784, 
0.9999, 
0.8717, 
0.8701, 
0.8782, 
0.8663, 
0.9999,  
0.9999, 
0.9999, 
0.7920, 
0.8547, 
0.9999, 
0.8882, 
0.8515, 
0.8723, 
1.0000, 
0.8592, 
0.9999, 
0.8556, 
0.8236, 
0.6042, 
0.8260, 
0.8384,])


birnn = np.array([0.6963, 
0.6499, 
0.4337, 
0.4032, 
0.3739, 
0.4750, 
0.5553, 
0.4839, 
0.5907, 
0.6547, 
0.5301, 
0.4779, 
0.6396, 
0.5730, 
0.7386, 
0.1317, 
0.1113, 
0.0564, 
0.5874, 
0.7207, 
0.6570, 
0.6411, 
0.6411, 
0.4220, 
0.5477, 
0.3993, 
0.2464, 
0.1698, 
0.2825, 
0.2825, 
0.4648, 
0.5024, 
0.5248, 
0.2148, 
0.1618, 
0.5891, 
0.6205, 
0.5365, 
0.4188, 
0.5472, 
0.5917, 
0.3057, 
0.4258, 
0.7346, 
0.6525,])


bertscore = np.array([1.0000,
0.8332,
0.8445,
0.8683,
1.0000,
0.8755,
0.8587,
1.0000,
0.8463,
0.8858,
1.0000,
0.9344,
1.0000,
0.8689,
0.8632,
1.0000,
0.8573,
0.8514,
0.9264,
0.8393,
0.8357,
0.8618,
0.8618,
1.0000,
0.8349,
0.8817,
0.8647,
0.8508,
1.0000,
1.0000,
0.9765,
0.8355,
0.8443,
1.0000,
0.8842,
0.9211,
0.8931,
1.0000,
0.8901,
1.0000,
0.8311,
0.8360,
0.8270,
0.8622,
0.8263,])

min_max = MinMaxScaler()
w2v_scaled = min_max.fit_transform(w2v.reshape(-1,1))
# print(w2v_scaled)

birnn_scaled = min_max.fit_transform(birnn.reshape(-1,1))
# print(birnn_scaled)

bertscore_scaled = min_max.fit_transform(bertscore.reshape(-1,1))
# print(bertscore_scaled)

avg_score = (w2v_scaled + birnn_scaled + bertscore_scaled)/3
# print(avg_score)

import matplotlib.pyplot as plt

human_scaled = [i[0] for i in human_scaled]
avg_score = [j[0] for j in avg_score]

a, b = np.polyfit(human_scaled, avg_score, 1)
x = np.linspace(0,1,101)
y = a*x + b

plt.scatter(human_scaled, avg_score)

corr, _ = pearsonr(human_scaled, avg_score)
print(corr)
plt.plot(x,y)
plt.xlabel("Human Score")
plt.ylabel("Automated Score")
plt.show()