import os

for filename in os.listdir("counterspeech_project-NLP\scores"):
   with open(os.path.join("counterspeech_project-NLP\scores", filename), 'r') as f:
       text = f.read().splitlines()
       print(filename)
       ans = sum([float(i) for i in text])
       print(ans/len(text))