import os

from parlai.scripts.interactive import Interactive
from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel
from parlai.scripts.display_model import DisplayModel
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
        
DisplayData.main(task="gab")

# DisplayModel.main(
#     task='gab',
#     model_file='zoo:pretrained_transformers/poly_model_huge_reddit/model',
#     num_examples=2,
# )

# TrainModel.main(
#     # similar to before
#     task='empathetic_dialogues', 
#     model='transformer/generator',
#     model_file='from_pretrained/model',
    
#     # initialize with a pretrained model
#     init_model='zoo:tutorial_transformer_generator/model',
    
#     # arguments we get from the pretrained model.
#     # Unfortunately, these must be looked up separately for each model.
#     n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
#     label_truncate=128, ffn_size=2048, embedding_size=512,
#     activation='gelu', variant='xlm',
#     dict_lower=True, dict_tokenizer='bpe',
#     dict_file='zoo:tutorial_transformer_generator/model.dict',
#     learn_positional_embeddings=True,
    
#     # some training arguments, specific to this fine-tuning
#     # use a small learning rate with ADAM optimizer
#     lr=1e-5, optimizer='adam',
#     warmup_updates=100,
#     # early stopping on perplexity
#     validation_metric='ppl',
#     # train at most 10 minutes, and validate every 0.25 epochs
#     max_train_time=600, validation_every_n_epochs=0.25,
    
#     # depend on your gpu. If you have a V100, this is good
#     batchsize=12, fp16=True, fp16_impl='mem_efficient',
    
#     # speeds up validation
#     skip_generation=True,
    
#     # helps us cram more examples into our gpu at a time
#     dynamic_batching='full',
# )