import os
import re
import pandas as pd

def remove_idx(idx):
    return float(re.sub(r".*\.\s", "", idx))

__location__ = os.getcwd()
df = pd.read_csv(f"{__location__}/RUBER/data/gab_scores_reference.txt", sep="\n", header=None)
print(df[0].apply(remove_idx).mean())

