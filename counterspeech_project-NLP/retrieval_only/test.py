import os

from parlai.scripts.display_model import DisplayModel
from parlai.scripts.eval_model import EvalModel

__location__ = os.getcwd()

# EvalModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
#     model_file='zoo:pretrained_transformers/poly_model_huge_reddit/model',
#     eval_candidates='fixed',
#     fixed_candidates_path=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
#     encode_candidate_vecs=True,
# )

# DisplayModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
#     model_file='zoo:pretrained_transformers/poly_model_huge_reddit/model', 
#     num_examples=10,
#     eval_candidates='fixed',
# )

EvalModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
    model_file=f'{__location__}/counterspeech_project-NLP/retrieval_only/from_pretrained_retrieval/model',
    eval_candidates='fixed',
    fixed_candidates_path=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
    encode_candidate_vecs=True,
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt",
    model_file=f'{__location__}/counterspeech_project-NLP/retrieval_only/from_pretrained_retrieval/model', 
    num_examples=30,
    eval_candidates='fixed',
    fixed_candidates_path=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/counter_speech_cand.txt",
    encode_candidate_vecs=True,
    verbose=True,
)