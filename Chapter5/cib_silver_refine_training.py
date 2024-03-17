import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd


dataset_name = 'lrml'

lrml_df = pd.read_csv('data/lrml_ds_v1.csv')


def get_splits(input_series, label_series, shuffle=False, test_size=0.1, max_length=512, tokenizer=None):
    dataset = transform_lists_to_ds(input_series.tolist(), label_series.tolist(), max_length=max_length,
                                    tokenizer=tokenizer)
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def get_datasets():
    prefix = 'parse English to LegalRuleML: '

    text = lrml_df['text']
    lrml = lrml_df['lrml']
    base_original_512_ds = get_splits(prefix + text, lrml, shuffle=False, max_length=512)

    return base_original_512_ds


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dr', '--decoding-runs', type=int, default='5',
                        help='How many decoding runs should be used?')
    parser.add_argument('-drp', '--decoding-runs-prediction', type=int,
                        help='How many decoding runs should be used?')
    parser.add_argument('--model-path', type=str, default='models/T5-AMR-lrml-silver-b4_0.0004_beam3_silver-gold_42/checkpoint-6000',
    # parser.add_argument('--model-path', type=str, default='models/T5-AMR-lrml-silver-b8_0.0002_beam3_silver_42/checkpoint-3750',
                        help='Model path to be used.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--write-predictions', action='store_true')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=4e-4,
                        help='The learning rate.')
    parser.add_argument('-bs', '--batchsize', type=int, default=8,
                        help='The batch size used for training. Evaluation uses batchsize * 2.')
    parser.add_argument('--num-beams', type=int, default=3,
                        help='Beam size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='The random seed.')
    parser.add_argument('--steps', type=int, default=250,
                        help='Evaluate after how many steps?')
    parser.add_argument('--eval', type=str, default='test',
                        help='On which dataset should be evaluated after training?')
    parser.add_argument('--name', type=str, default='silver-refine-sg',
                        help='The experiment name, what makes this run special?')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Which gpu should it run on')
    parser.add_argument('--epochs', type=int, default=20,
                        help='The random seed.')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lr = args.learning_rate
    bs = args.batchsize
    
    # lrs = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4]
    lrs = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]
    bss = [4, 8]
    #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
    from lrml_train_pred import transform_lists_to_ds, train

    for seed in [42]:
        for lr in lrs:
            for bs in bss:
                exp_name = 'b' + str(bs) + '_' + str(lr) + '_beam' + str(args.num_beams) + '_' + args.name + '_' + str(seed)

                hyperparameters = dict(epochs=args.epochs, learning_rate=lr, batch_size=bs, adafactor=False,
                                       decay=1.000e-4, warmup_steps=100, dataset=dataset_name, model_name='T5-AMR',
                                       model_path=args.model_path, run_name=exp_name, eval_strategy='epoch',
                                       eval_steps=args.steps, early_stopping_patience=3, max_length=512,
                                       seed=seed, write_predictions=args.write_predictions, evaluate_only=args.evaluate_only,
                                       evaluation_set=args.eval, num_beams=args.num_beams
                                       )
                print(hyperparameters)
                try:
                    train(datasets=get_datasets(), hyperparameters=hyperparameters)
                except Exception as e:
                    print(e)
