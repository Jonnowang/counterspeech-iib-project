import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gab = pd.read_csv('counterspeech_project-NLP/data/gab.csv')
print(gab["text"][:3])

char_len = gab['text'].str.len()
print(char_len.idxmax())
plt.show()