from bert_score import score, plot_example
import os
import pandas as pd
import matplotlib.pyplot as plt

__location__ = os.getcwd()

file_names = [f"{__location__}/RUBER/output_data/retrieval_response.txt",]

save_names = [f"{__location__}/RUBER/scores/retrieval_bertscore.txt",]

refs = pd.read_csv(f"{__location__}/RUBER/output_data/human_response.txt", sep="\n", header=None)
refs = refs[0].tolist()
refs = [str(i) for i in refs]
print(len(refs))

for filename, savename in zip(file_names,save_names):
    cands = pd.read_csv(filename, sep="\n", header=None)
    cands = cands[0].tolist()
    cands = [str(i) for i in cands]
    print(len(cands))
    print(cands[0])

    P, R, F1 = score(cands, refs, lang='en', verbose=True, idf=False, rescale_with_baseline=True)

    with open(savename, 'w') as f:
        for s in F1:
            print(s.item())
            f.write(str(s.item())+"\n")
    f.close()