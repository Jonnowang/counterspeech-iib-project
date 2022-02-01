import argparse
import os
import logging

from hybrid_evaluation import Hybrid


# todo: add option to obtain ref and unref scores.
# todo: add option to make combiner of ref and unref changeable.

__location__ = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', default=f"{__location__}/RUBER/train")
    parser.add_argument('-query_max_len', default=60, type=int, help='max length of query')
    parser.add_argument('-reply_max_len', default=60, type=int, help='max length of reply')
    parser.add_argument('-w2v_file', default=f"{__location__}/RUBER/data/w2v_1mil.txt", help='word2vec in text format')
    parser.add_argument('-pooling_type', default='all', choices=(
        'max_min',
        'avg',
        'all',
    ), help='how to compute sentence vector from word vectors in the ref metric (pooling)')
    parser.add_argument('-query_w2v_file', help='query embedding file', default=f"{__location__}/RUBER/output/gab_query_short.txt.embed")
    parser.add_argument('-reply_w2v_file', help='reply embedding file', default=f"{__location__}/RUBER/output/gab_response_short.txt.embed")
    parser.add_argument('-query_file', default=f"{__location__}/RUBER/data/gab_query_short.txt")
    parser.add_argument('-reply_file', default=f"{__location__}/RUBER/data/gab_response_short.txt")
    parser.add_argument('-query_vocab_file', default=f"{__location__}/RUBER/output/gab_query_short.txt.vocab60")
    parser.add_argument('-reply_vocab_file', default=f"{__location__}/RUBER/output/gab_response_short.txt.vocab60")
    parser.add_argument('-generated_file', default=f"{__location__}/RUBER/data/gab_response_short_test.txt")
    parser.add_argument('-score_file', default=f"{__location__}/RUBER/data/gab_scores_trial.txt", help='scores stored here')
    parser.add_argument('-type', choices=('ruber', 'ref', 'unref'), default='ref', help='type of metric to compute')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    model = Hybrid(
        word2vec_file=args.w2v_file,
        query_w2v_file=args.query_w2v_file,
        reply_w2v_file=args.reply_w2v_file,
        train_dir=args.train_dir,
        query_max_len=args.query_max_len,
        reply_max_len=args.reply_max_len,
        pooling_type=args.pooling_type,
    )

    if args.type == 'ruber':
        scores = model.get_scores(
            query_file=args.query_file,
            reply_file=args.reply_file,
            generated_file=args.generated_file,
            query_vocab_file=args.query_vocab_file,
            reply_vocab_file=args.reply_vocab_file,
        )
    elif args.type == 'ref':
        scores = model.get_ref_scores(
            reply_file=args.reply_file,
            generated_file=args.generated_file,
        )
    else:
        scores = model.get_unref_scores(
            query_file=args.query_file,
            generated_file=args.generated_file,
            query_vocab_file=args.query_vocab_file,
            reply_vocab_file=args.reply_vocab_file,
        )

    for i in range(len(scores)):
        with open(args.score_file, 'a') as f:
            f.write(f'{i+1}. {scores[i]}\n')