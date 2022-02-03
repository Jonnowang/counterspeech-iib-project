import os

from parlai.scripts.interactive import Interactive
from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel
from parlai.scripts.display_model import DisplayModel
from parlai.core.teachers import register_teacher, DialogTeacher

__location__ = os.getcwd()
# with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_query.txt") as fq:
#     queries = fq.readlines()
# with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_response.txt") as fr:
#     responses = fr.readlines()

# for query, response in zip(queries[:7000], responses[:7000]):
#     with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_train.txt", 'a') as f:
#         f.write(f"text:{query.rstrip()}\tlabels:{response.rstrip()}\tepisode_done:{True}\n")

# for query, response in zip(queries[7000:9000], responses[7000:9000]):
#     with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_valid.txt", 'a') as f:
#         f.write(f"text:{query.rstrip()}\tlabels:{response.rstrip()}\tepisode_done:{True}\n")

# for query, response in zip(queries[9000:], responses[9000:]):
#     with open(f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data_test.txt", 'a') as f:
#         f.write(f"text:{query.rstrip()}\tlabels:{response.rstrip()}\tepisode_done:{True}\n")

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
        
DisplayData.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data",
    fromfile_datatype_extension=True,
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data",
    fromfile_datatype_extension=True,
    model_file='zoo:pretrained_transformers/poly_model_huge_reddit/model',
    num_examples=10,
    eval_candidates='batch',
)

TrainModel.main(
    # similar to before
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data",
    fromfile_datatype_extension=True,
    model='transformer/polyencoder',
    model_file=f'{__location__}/counterspeech_project-NLP/retrieval_only/from_pretrained_retrieval/model',
    
    # initialize with a pretrained model
    init_model='zoo:pretrained_transformers/poly_model_huge_reddit/model',
    dict_file='zoo:pretrained_transformers/poly_model_huge_reddit/model.dict',
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    lr_scheduler_patience=0, lr_scheduler_decay=0.4,
    data_parallel=True, history_size=20, label_truncate=72,
    text_truncate=360, veps=0.5, vme=8000,
    save_after_valid=True, log_every_n_secs=20, candidates='batch',
    dict_tokenizer='bpe', dict_lower=True,
    variant='xlm', reduction_type='mean', share_encoders=False,
    learn_positional_embeddings=True, n_layers=12, n_heads=12,
    ffn_size=3072, attention_dropout=0.1, relu_dropout=0.0,
    dropout=0.1, n_positions=1024, embedding_size=768,
    activation='gelu', embeddings_scale=False, n_segments=2,
    learn_embeddings=True, polyencoder_type='codes',
    poly_n_codes=64, poly_attention_type='basic',
    dict_endtoken='__start__', eval_candidates='batch',

    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=5e-5, optimizer='adamax',
    warmup_updates=100, output_scaling=0.06,
    # early stopping on perplexity
    validation_metric='accuracy',
    validation_metric_mode='max',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=43200, validation_every_n_epochs=0.25, num_epochs=8.0,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=32, eval_batchsize=10, fp16=True,
)

DisplayModel.main(
    task="fromfile:parlaiformat",
    fromfile_datapath=f"{__location__}/counterspeech_project-NLP/retrieval_only/data/gab_data",
    fromfile_datatype_extension=True,
    force_fp16_tokens=True,
    model_file=f'{__location__}/counterspeech_project-NLP/retrieval_only/from_pretrained_retrieval/model', 
    num_examples=10,
)