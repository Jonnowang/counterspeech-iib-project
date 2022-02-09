import os

from parlai.scripts.display_data import DisplayData
from parlai.scripts.display_model import DisplayModel
from parlai.scripts.multiprocessing_train import MultiProcessTrain
from parlai.scripts.multiprocessing_eval import MultiProcessEval
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
    model_file='zoo:blender/reddit_3B/model',
    num_examples=10,
)

# --multitask-weights 1,3,3,3 -veps 0.25 --attention-dropout 0.0
# --model transformer/generator --embedding-size 2560 --ffn-size 10240 --variant prelayernorm 
# --n-heads 32 --n-positions 128 --n-encoder-layers 2 --n-decoder-layers 24 --history-add-global-end-token end 
# --delimiter '  ' --dict-tokenizer bytelevelbpe  --dropout 0.1
# --label-truncate 128 --log_every_n_secs 10
# --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --relu-dropout 0.0 --activation gelu 
# --model-parallel true --save-after-valid True --text-truncate 128 --truncate 128 --warmup_updates 100 
# --fp16-impl mem_efficient --update-freq 2 --gradient-clip 0.1 --skip-generation True -vp 10 -vmt ppl -vmm min 
# --model-file /tmp/test_train_27B

MultiProcessTrain.main(
    # similar to before
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data",
    fromfile_datatype_extension=True,
    model='transformer/generator',
    model_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative/model',
    
    # initialize with a pretrained model
    init_model='zoo:blender/reddit_3B/model',
    dict_file='zoo:blender/reddit_3B/model.dict',
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    multitask_weights=[1,3,3,3], veps=0.25, attention_dropout=0.0,
    embedding_size=2560, ffn_size=10240, variant='prelayernorm',
    n_heads=32, n_positions=128, n_encoder_layers=2, n_decoder_layers=24,
    history_add_global_end_token='end', delimiter='  ', dict_tokenizer='bytelevelbpe',
    dropout=0.1, label_trucate=128, log_every_n_secs=10,
    lr_scheduler="reduceonplateau", lr_scheduler_patience=3,
    relu_dropout=0.0, activation='gelu', model_parallel=True,
    save_after_valid=True, text_truncate=128, truncate=128,
    update_freq=2, gradient_clip=0.1, skip_generation=True, vp=10,
    vmt='ppl', vmm='min',

    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=7e-06, optimizer='adam',
    warmup_updates=100,
    # early stopping on perplexity
    validation_metric='accuracy',
    validation_metric_mode='max',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=43200, validation_every_n_epochs=0.25, num_epochs=8.0,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=32, eval_batchsize=10, fp16=True,
)

# Evaluate Base Model
MultiProcessEval.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    model_file='zoo:blender/reddit_3B/model',
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data_test.txt",
    force_fp16_tokens=True,
    model_file='zoo:blender/reddit_3B/model', 
    num_examples=10,
)

# Evaluate Fine Tuned Model
MultiProcessEval.main(
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