import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd


dataset_name = 'lrml'

# LRML Dataset with all rules and all data cleansing steps applied
lrml_df_v7 = pd.read_csv('data/lrml_ds_v7.csv')

doc_df = pd.read_csv('data/lrml_ds_doc-exp.csv')


def get_random_split_sample(df, valid_size):
    prefix = 'parse English to LegalRuleML: '
    text, lrml = df['text'], df['lrml']
    text = prefix + text
    train_df = df.sample(valid_size)
    valid_df =df.loc[~df['logic_6'].isin(train_df['logic_6'])]
    train_ds = transform_lists_to_ds(text[train_df.index].tolist(), lrml[train_df.index].tolist())
    valid_ds = transform_lists_to_ds(text[valid_df.index].tolist(), lrml[valid_df.index].tolist())
    return {'train': train_ds, 'test': valid_ds}


def get_datasets():
    datasets = {}

    datasets['random_pre_600'] = get_random_split_sample(doc_df, 518)
    datasets['random_pre_760'] = get_random_split_sample(lrml_df_v7, 684)

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
