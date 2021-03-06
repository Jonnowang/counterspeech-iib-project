import argparse
import os

import data_helpers

__location__ = os.getcwd()

def make_data(args, max_len, raw_input):
    vocab_file, _, _ = data_helpers.process_train_file(
        data_dir=args.data_dir,
        raw_input=raw_input,
        max_length=max_len,
        min_frequency=args.min_freq,
    )

    w2v, vec_dim, _ = data_helpers.load_word2vec(args.w2v_file)
    data_helpers.make_embedding_matrix(
        data_dir=args.data_dir,
        prefix=os.path.basename(raw_input),
        word2vec=w2v,
        vec_dim=vec_dim,
        vocab_file=vocab_file,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default=f"{__location__}/RUBER/output", help='where the output data will be stored')
    parser.add_argument('-w2v_file', default=f"{__location__}/RUBER/data/w2v_1mil.txt", help='word2vec in text format')
    parser.add_argument('-min_freq', type=int, default=0, help='min frequency of word')
    parser.add_argument('-query_max_len', default=60, type=int, help='max length of query')
    parser.add_argument('-reply_max_len', default=60, type=int, help='max length of reply')
    parser.add_argument('-query_file', default=f"{__location__}/RUBER/output_data/inputs.txt")
    parser.add_argument('-reply_file', default=f"{__location__}/RUBER/output_data/human_response.txt")
    args = parser.parse_args()

    make_data(args, args.query_max_len, args.query_file)
    make_data(args, args.reply_max_len, args.reply_file)
