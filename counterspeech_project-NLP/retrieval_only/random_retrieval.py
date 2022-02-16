import os

from parlai.scripts.display_model import DisplayModel
from parlai.scripts.eval_model import EvalModel
from parlai.scripts.build_candidates import BuildCandidates
from parlai.scripts.build_dict import BuildDict

__location__ = os.getcwd()

# EvalModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
#     model='random_candidate',
#     label_candidates_file=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
# )

# DisplayModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
#     model='random_candidate', 
#     num_examples=10,
#     label_candidates_file=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
# )

BuildDict.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_train.txt",
    dict_file=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab.dict",
)