import pandas as pd
import re
import matplotlib.pyplot as plt

# with open(r'counterspeech_project-NLP\base3B.txt', 'r') as file:
#     data = file.read(encoding="utf8")

file = open(r'counterspeech_project-NLP\data\greedy_double.txt', errors="ignore")
data = file.read()

outputs = re.findall(r"\[text\]: .*", data)
# print(outputs)
# for output in outputs:
#     print(output[7:])

with open(r'counterspeech_project-NLP\data\greedy_double_response.txt', "w") as f:
    i = False
    for output in outputs:
        if i:
            f.write(output[8:] + "\n")
            i = not i
        else:
            i = not i
f.close()