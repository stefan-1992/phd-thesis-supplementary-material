import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from lrml_utils import fix_lrml_tokenisation, add_title_reference, \
    apply_all_canges, define_fix, type_fix, remove_duplicate_and_or



dataset_name = 'lrml'

# LRML Dataset with improved alignments - Step 1
lrml_df_v3 = pd.read_csv('data/lrml_ds_v3.csv')
# LRML Dataset with tacit knowledge added to the input clauses - Step 2
lrml_df_v4 = pd.read_csv('data/lrml_ds_v4.csv')
# LRML Dataset with LRML rules cleansed - Steps 3-5
lrml_df_v5 = pd.read_csv('data/lrml_ds_v5.csv')
# LRML Dataset with manual reference cleansing - Base for Reference and Entity cleaning
lrml_df_v6 = pd.read_csv('data/lrml_ds_v6.csv')
# LRML Dataset with all rules and all data cleansing steps applied
lrml_df_v7 = pd.read_csv('data/lrml_ds_v7.csv')



def get_splits(input_series, label_series, shuffle=False, test_size=0.1, max_length=1024):
    dataset = transform_lists_to_ds(input_series.tolist(), label_series.tolist(), max_length=max_length)
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def get_doc_split(df, text, lrml):
    train_index = df.loc[~(df['file'].str.contains('G14VM1')) & ~(df['file'].str.contains('D1AS1')) & ~(
        df['file'].str.contains('B2AS1'))].index
    valid_index = df.loc[(df['file'].str.contains('G14VM1')) | (df['file'].str.contains('D1AS1')) | (
        df['file'].str.contains('B2AS1'))].index
    train_ds = transform_lists_to_ds(text[train_index].tolist(), lrml[train_index].tolist())
    valid_ds = transform_lists_to_ds(text[valid_index].tolist(), lrml[valid_index].tolist())
    return {'train': train_ds, 'test': valid_ds}



def get_datasets():
    prefix = 'parse English to LegalRuleML: '
    datasets = {}

    text = lrml_df_v3['text']
    lrml = lrml_df_v3['lrml']
    datasets['clean2_alignment_ds'] = get_doc_split(lrml_df_v3, prefix + text, lrml)

    text = lrml_df_v4['text']
    lrml = lrml_df_v4['lrml']
    datasets['clean2_add_tacit'] = get_doc_split(lrml_df_v4, prefix + text, lrml)

    text = lrml_df_v5['text']
    lrml = lrml_df_v5['lrml']
    datasets['clean2_lrml_phase_1'] = get_doc_split(lrml_df_v5, prefix + text, lrml)

    text = lrml_df_v6['text']
    lrml = lrml_df_v6.apply(add_title_reference, axis=1)
    lrml = fix_lrml_tokenisation(lrml)
    datasets['clean2_lrml_phase_2_ref'] = get_doc_split(lrml_df_v6, prefix + text, lrml)

    text = lrml_df_v6['text']
    lrml = lrml_df_v6.apply(add_title_reference, axis=1)
    lrml = apply_all_canges(lrml).apply(define_fix).apply(type_fix).apply(remove_duplicate_and_or)
    lrml = fix_lrml_tokenisation(lrml)
    datasets['clean2_lrml_phase_2_entity'] = get_doc_split(lrml_df_v6, prefix + text, lrml)

    text = lrml_df_v7['text']
    lrml = lrml_df_v7['lrml']
    datasets['clean2_lrml_all'] = get_doc_split(lrml_df_v7, prefix + text, lrml)

    return datasets


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script", formatter_class=ArgumentDefaultsHelpFormatter)

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
    exp_name = 'b' + str(bs) + '_' + str(lr) + '_' + args.name + '_' + str(args.seed)

    if args.evaluate_only:
        exp_name += '_eval'

    #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
    from lrml_train_pred import transform_lists_to_ds, train

    for dataset_name, ds in get_datasets().items():
        hyperparameters = dict(epochs=args.epochs, learning_rate=lr, batch_size=bs, adafactor=False, decay=1.000e-4,
                               warmup_steps=100, dataset=dataset_name, model_name='T5-AMR', model_path=args.model_path,
                               run_name=exp_name, eval_strategy='epoch', eval_steps=args.steps,
                               early_stopping_patience=5, max_length=1024,
                               seed=args.seed, write_predictions=args.write_predictions,
                               evaluate_only=args.evaluate_only, evaluation_set=args.eval, num_beams=args.num_beams
                               )
        print(hyperparameters)

        train(datasets=ds, hyperparameters=hyperparameters)
