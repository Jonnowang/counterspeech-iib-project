import argparse
import os
from unreferenced_metric import Unreferenced

GRU_NUM_UNITS = 128
MLP_UNITS = (256, 512, 128)

__location__ = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', default=f"{__location__}/RUBER/train")
    parser.add_argument('-query_file', default=f"{__location__}/RUBER/output/gab_query_short.txt.id60")
    parser.add_argument('-reply_file', default=f"{__location__}/RUBER/output/gab_response_short.txt.id60")
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-checkpoint_freq', help='checkpoint frequency', type=int, default=400)
    parser.add_argument('-query_max_len', default=60, type=int, help='max length of query')
    parser.add_argument('-reply_max_len', default=60, type=int, help='max length of reply')
    parser.add_argument('-query_embed_file', help='query embedding file', default=f"{__location__}/RUBER/output/gab_query_short.txt.embed")
    parser.add_argument('-reply_embed_file', help='reply embedding file', default=f"{__location__}/RUBER/output/gab_response_short.txt.embed")
    args = parser.parse_args()

    model = Unreferenced(
        train_dir=args.train_dir,
        query_max_len=args.query_max_len,
        reply_max_len=args.reply_max_len,
        query_w2v_file=args.query_embed_file,
        reply_w2v_file=args.reply_embed_file,
        gru_num_units=GRU_NUM_UNITS,
        mlp_units=MLP_UNITS,
    )

    model.train(
        query_file=args.query_file,
        reply_file=args.reply_file,
        batch_size=args.batch_size,
        steps_per_checkpoint=args.checkpoint_freq
    )
