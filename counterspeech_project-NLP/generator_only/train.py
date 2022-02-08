import os

from parlai.scripts.interactive import Interactive
from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel
from parlai.scripts.display_model import DisplayModel
from parlai.scripts.eval_model import EvalModel
from parlai.core.teachers import register_teacher, DialogTeacher

__location__ = os.getcwd()

@register_teacher("gab")
class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # opt is the command line arguments.
        
        # What is this shared thing?
        # We make many copies of a teacher, one-per-batchsize. Shared lets us store 
        
        # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
        # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
        # the fold name (train/valid/test) + ".txt"
        opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
        super().__init__(opt, shared)
    
    def setup_data(self, datafile):
        # filename tells us where to load from.
        # We'll just use some hardcoded data, but show how you could read the filename here:
        print(f" ~~ Loading from {datafile} ~~ ")
        
        # setup_data should yield tuples of ((text, label), new_episode)
        # That is ((str, str), bool)
        
        with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_query.txt") as fq:
            queries = fq.readlines()
        with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_response.txt") as fr:
            responses = fr.readlines()
        
        for query, response in zip(queries, responses):
            yield (query, response), True
        
DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data",
    fromfile_datatype_extension=True,
    model_file='zoo:blenderbot2/blenderbot2_3B/model',
    num_examples=10,
    eval_candidates='batch',
)

TrainModel.main(
    # similar to before
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data",
    fromfile_datatype_extension=True,
    model='projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent',
    model_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative/model',
    
    # initialize with a pretrained model
    init_model='zoo:blenderbot2/blenderbot2_3B/model',
    dict_file='zoo:blenderbot2/blenderbot2_3B/model.dict',
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    gpu=-1, data_parallel=False, gradient_clip=0.1,
    adam_eps=1e-08, nesterov=True, nus=[0.7],
    betas=[0.9, 0.999], update_freq=1,
    no_cuda=False,

    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=5e-5, optimizer='mem_eff_adam',
    warmup_updates=100, warmup_rate=0.0001,
    # early stopping on perplexity
    validation_metric='accuracy',
    validation_metric_mode='max',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=43200, validation_every_n_epochs=0.25, num_epochs=8.0,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=32, eval_batchsize=10, fp16=True,
)

# Evaluate Base Model
EvalModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    model_file='zoo:blenderbot2/blenderbot2_3B/model',
    eval_candidates='batch',
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    force_fp16_tokens=True,
    model_file='zoo:blenderbot2/blenderbot2_3B/model', 
    num_examples=10,
    eval_candidates='batch',
)

# Evaluate Fine Tuned Model
EvalModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    model_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative/model', 
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    force_fp16_tokens=True,
    model_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative/model', 
    num_examples=10,
)