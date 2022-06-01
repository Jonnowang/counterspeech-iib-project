import numpy as np
import pandas as pd
import re

df = pd.read_csv("/home/jonathan/IIB_MEng_Project/all_data/Counter Speech Selection Survey.csv")

for col in df.columns:
    m = re.search(r"Response:\s.*",col)
    if m is not None:
        s = m.group(0)
        print(s[10:])