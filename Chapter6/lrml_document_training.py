import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd


dataset_name = 'lrml'

# LRML Dataset with all rules and all data cleansing steps applied
lrml_df_v7 = pd.read_csv('data/lrml_ds_v7.csv')

# Doc df was created as baseline for the document experiments with the following two line
# included_df = lrml_df_v7.loc[(lrml_df_v7['file'].str.contains('E2AS1'))].sample(82)
# doc_df = lrml_df_v7.loc[~(lrml_df_v7['file'].str.contains('E2AS1'))].append(included_df)
doc_df = pd.read_csv('data/lrml_ds_doc-exp.csv')



def get_doc_split_sample(df, files):
    prefix = 'parse English to LegalRuleML: '
    text, lrml = df['text'], df['lrml']
    text = prefix + text
    train_df = df
    valid_df = None
    for i in files:
        train_df = train_df.loc[~(train_df['file'].str.contains(i))]
        if valid_df is not None:
            valid_df = valid_df.append(df.loc[(df['file'].str.contains(i))])
        else:
            valid_df = df.loc[(df['file'].str.contains(i))]
    train_df = train_df.sample(518)
    train_ds = transform_lists_to_ds(text[train_df.index].tolist(), lrml[train_df.index].tolist())
    valid_ds = transform_lists_to_ds(text[valid_df.index].tolist(), lrml[valid_df.index].tolist())
    return {'train': train_ds, 'test': valid_ds}


def get_datasets():

    datasets = {}
    documents = {'B1AS1': ['B1AS1'], 'E1AS1': ['E1AS1'], 'B1AS3': ['B1AS3'], 'CAS2': ['CAS2'], 'G12AS2': ['G12AS2'],
                 'G13AS2': ['G13AS2'], 'G12AS1': ['G12AS1'], 'G14VM1': ['G14VM1'], 'G13AS1': ['G13AS1'],
                 'G15AS1': ['G15AS1'], 'D1AS1': ['D1AS1'], 'B2AS1': ['B2AS1'],
                 'others': ['E3AS1', 'G14AS1', 'G1AS1' 'G4AS1', 'CVM']}

    for i, j in documents.items():
        datasets['document_' + i] = get_doc_split_sample(doc_df, j)

    datasets['document_' + 'baseline'] = get_doc_split_sample(doc_df, ['G14VM1', 'D1AS1', 'B2AS1'])
    datasets['document_' + 'E2AS1'] = get_doc_split_sample(lrml_df_v7, ['E2AS1'])

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
