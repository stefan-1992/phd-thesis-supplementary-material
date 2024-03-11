import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

dataset_name = 'lrml'

lrml_df = pd.read_csv('data/lrml_ds_v1.csv')


def get_splits(input_series, label_series, shuffle=False, test_size=0.1, max_length=512, tokenizer=None):
    dataset = transform_lists_to_ds(input_series.tolist(), label_series.tolist(), max_length=max_length,
                                    tokenizer=tokenizer)
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def get_datasets(tokenizer):
    prefix = 'parse English to LegalRuleML: '

    text = lrml_df['text']
    lrml = lrml_df['lrml']
    base_original_512_ds = get_splits(prefix + text, lrml, shuffle=False, max_length=512, tokenizer=tokenizer)

    return base_original_512_ds

MODEL_T5_BASE = 't5-base'


def get_tokenizer_and_model(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, default='t5-base',
                        help='Model path to be used.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--write-predictions', action='store_true')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=5e-4,
                        help='The learning rate.')
    parser.add_argument('-bs', '--batchsize', type=int, default=8,
                        help='The batch size used for training. Evaluation uses batchsize * 2.')
    parser.add_argument('--num-beams', type=int, default=3,
                        help='Beam size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='The random seed.')
    parser.add_argument('--steps', type=int, default=500,
                        help='Evaluate after how many steps?')
    parser.add_argument('--eval', type=str, default='test',
                        help='On which dataset should be evaluated after training?')
    parser.add_argument('--name', type=str, default='hyper',
                        help='The experiment name, what makes this run special?')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Which gpu should it run on')
    parser.add_argument('--epochs', type=int, default=30,
                    help='The random seed.')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lr = args.learning_rate
    bs = args.batchsize
    beam_num = args.num_beams

    model_path = MODEL_T5_BASE
    
    #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from lrml_train_pred import transform_lists_to_ds, train

    exp_name = 'b' + str(bs) + '_' + str(lr) + '_beam' + str(beam_num) + '_' + args.name + '_' + str(args.seed)

    hyperparameters = dict(epochs=args.epochs, learning_rate=lr, batch_size=bs, adafactor=False,
                           decay=1.000e-4, warmup_steps=100, dataset=dataset_name, model_name=args.model_path,
                           model_path=model_path, run_name=exp_name, eval_strategy='epoch',
                           eval_steps=args.steps, early_stopping_patience=5, max_length=512,
                           seed=args.seed, write_predictions=args.write_predictions, evaluate_only=args.evaluate_only,
                           evaluation_set=args.eval, num_beams=beam_num
                           )
    print(hyperparameters)
    tokenizer, model = get_tokenizer_and_model(model_path)
    model = T5ForConditionalGeneration(model.config)
    ds = get_datasets(tokenizer)
    try:
        train(datasets=ds, hyperparameters=hyperparameters, model=model,
              tokenizer=tokenizer)
    except Exception as e:
        print(e)
