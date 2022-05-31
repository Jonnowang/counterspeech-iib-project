import os

from parlai.scripts.display_model import DisplayModel
from parlai.scripts.eval_model import EvalModel

__location__ = os.getcwd()


# # Evaluate Base Model
# EvalModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
#     model_file='zoo:blender/reddit_3B/model',
# )

# DisplayModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
#     force_fp16_tokens=True,
#     model_file='zoo:blender/reddit_3B/model', 
#     num_examples=2169,
#     skip_generation=False,
# )


# # Evaluate Fine Tuned Model
# EvalModel.main(
#     task="fromfile:parlaiformat",
#     fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
#     model_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative_conan_temperature/model', 
# )

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    model='transformer/generator',
    model_file=f'{__location__}/counterspeech_project-NLP/generator_only/generative_greedy_double/model', 
    num_examples=2169,
    skip_generation=False,
    verbose=True,
)