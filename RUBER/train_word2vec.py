# Train and save word2vec on the target corpus.
# The pretrained one suffer from OOV problem (especially on Ubuntu).

import os
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

if __name__ == '__main__':
    __location__ = os.getcwd()

    print(common_texts)
    model = Word2Vec.load(f"{__location__}/RUBER/data/w2v_100d.txt")
    model.save(f"{__location__}/RUBER/data/word2vec.model")

