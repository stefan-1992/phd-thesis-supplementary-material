import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

from lrml_utils import tree_based_spacing, reverse_loop, reverse_resolve_expressions, reverse_combine_rel_and_var, reverse_move_and_or_to_data_node, reverse_units, resolve_expressions


dataset_name = 'lrml'

lrml_df = pd.read_csv('data/lrml_ds_v8.csv')


def get_splits(input_series, label_series, shuffle=False, test_size=0.1, max_length=1024):
    dataset = transform_lists_to_ds(
        input_series.tolist(), label_series.tolist(), max_length=max_length)
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def get_doc_split(df, text, lrml):
    train_index = df.loc[~(df['file'].str.contains('G14VM1')) & ~(df['file'].str.contains('D1AS1')) & ~(
        df['file'].str.contains('B2AS1'))].index
    valid_index = df.loc[(df['file'].str.contains('G14VM1')) | (df['file'].str.contains('D1AS1')) | (
        df['file'].str.contains('B2AS1'))].index
    train_ds = transform_lists_to_ds(
        text[train_index].tolist(), lrml[train_index].tolist())
    valid_ds = transform_lists_to_ds(
        text[valid_index].tolist(), lrml[valid_index].tolist())
    return {'train': train_ds, 'test': valid_ds}


def get_file_names_from_df(df):
    return df['file'].str.split('-').apply(lambda x: x[1]).str.split('#').apply(lambda x: x[0])


def get_text_and_lrml(df):
    text = 'parse English to LegalRuleML: ' + \
        get_file_names_from_df(df) + ' ' + df['text']
    lrml = df['lrml'].apply(lambda x: get_original(
        x)).apply(tree_based_spacing)
    return text, lrml


def get_original(lrml):
    prefix = ''
    lrml = reverse_loop(lrml, prefix=prefix)
    lrml = reverse_resolve_expressions(lrml, fix_errors=True, prefix=prefix)
    lrml = reverse_combine_rel_and_var(lrml, prefix=prefix)
    if 'atom(' in lrml:
        lrml = reverse_move_and_or_to_data_node(lrml)
    else:
        lrml = resolve_expressions(reverse_move_and_or_to_data_node(
            reverse_resolve_expressions(lrml, fix_errors=True, prefix=prefix)))
    lrml = reverse_units(lrml, prefix=prefix)
    return lrml


def get_datasets():
    text, lrml = get_text_and_lrml(lrml_df)
    ir_paper_baseline = get_doc_split(lrml_df, text, lrml)

    ds_name = 'ir_paper_baseline'
    return {ds_name: ir_paper_baseline}


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-path', type=str, default='models/model_t5-amr/',
                        help='Model path to be used.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--write-predictions', action='store_true')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4,
                        help='The learning rate.')
    parser.add_argument('-bs', '--batchsize', type=int, default=4,
                        help='The batch size used for training. Evaluation uses batchsize * 2.')
    parser.add_argument('--num-beams', type=int, default=5,
                        help='Beam size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='The random seed.')
    parser.add_argument('--steps', type=int, default=500,
                        help='Evaluate after how many steps?')
    parser.add_argument('--eval', type=str, default='test',
                        help='On which dataset should be evaluated after training?')
    parser.add_argument('--name', type=str, required=True,
                        help='The experiment name, what makes this run special?')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Which gpu should it run on')
    parser.add_argument('--epochs', type=int, default=30,
                        help='The random seed.')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lr = args.learning_rate
    bs = args.batchsize
    exp_name = 'b' + str(bs) + '_' + str(lr) + '_' + \
        args.name + '_' + str(args.seed)

    if args.evaluate_only:
        exp_name += '_eval'

    #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
    from lrml_train_pred import transform_lists_to_ds, train

    for dataset_name, ds in get_datasets().items():
        for seed in [42, 43, 44]:
            hyperparameters = dict(epochs=args.epochs, learning_rate=lr, batch_size=bs, adafactor=False, decay=1.000e-4,
                                   warmup_steps=100, dataset=dataset_name, model_name='T5-AMR', model_path=args.model_path,
                                   run_name=exp_name, eval_strategy='epoch', eval_steps=args.steps,
                                   early_stopping_patience=5, max_length=1024,
                                   seed=seed, write_predictions=args.write_predictions,
                                   evaluate_only=args.evaluate_only, evaluation_set=args.eval, num_beams=args.num_beams
                                   )
            print(hyperparameters)


            train(datasets=ds, hyperparameters=hyperparameters)
