import os

from parlai.scripts.train_model import TrainModel
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
        
TrainModel.main(
    # similar to before
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/generator_only/data/gab_data",
    fromfile_datatype_extension=True,
    model='transformer/generator',
    model_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative_conan_temperature/model',
    
    # initialize with a pretrained model
    init_model=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative_2/model',
    dict_file=f'{__location__}/counterspeech_project-NLP/generator_only/from_pretrained_generative_2/model.dict',
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    multitask_weights=[1,3,3,3], veps=0.25, attention_dropout=0.0,
    embedding_size=2560, ffn_size=10240, variant='prelayernorm',
    n_heads=32, n_positions=128, n_encoder_layers=2, n_decoder_layers=24,
    history_add_global_end_token='end', delimiter='  ', dict_tokenizer='bytelevelbpe',
    dropout=0.1, label_truncate=128, log_every_n_secs=10,
    lr_scheduler="reduceonplateau", lr_scheduler_patience=3,
    relu_dropout=0.0, activation='gelu', model_parallel=True,
    save_after_valid=True, text_truncate=128, truncate=128,
    update_freq=2, gradient_clip=0.1, skip_generation=True, vp=10,
    vmt='ppl', vmm='min', topk=25, temperature=4.0,

    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=7e-06, optimizer='adam',
    warmup_updates=100,
    # early stopping on perplexity
    validation_metric='ppl',
    validation_metric_mode='max',
    # train at most 10 minutes, and validate every epoch
    max_train_time=43200, validation_every_n_epochs=1.0, num_epochs=10.0,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=32, eval_batchsize=10, fp16=True,

    beam_size=20, inference='beam',
)