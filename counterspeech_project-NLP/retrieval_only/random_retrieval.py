import os

from parlai.scripts.interactive import Interactive
from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel
from parlai.scripts.display_model import DisplayModel
from parlai.scripts.eval_model import EvalModel
from parlai.core.teachers import register_teacher, DialogTeacher

__location__ = os.getcwd()

EvalModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
    model='random_candidate',
    label_candidates_file=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
    model='random_candidate', 
    num_examples=3,
    label_candidates_file=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
)