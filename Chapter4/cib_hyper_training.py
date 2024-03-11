import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

MODEL_T5_BASE = 't5-base'
MODEL_T5_AMR_PATH = '/path-to/model_t5-amr/'
MODEL_T5_LARGE = 't5-large'
MODEL_BART_BASE = 'facebook/bart-base'
MODEL_BART_LARGE = 'facebook/bart-large'

model_names = [MODEL_T5_BASE, MODEL_T5_AMR_PATH, MODEL_T5_LARGE, MODEL_BART_BASE, MODEL_BART_LARGE]
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


def get_tokenizer_and_model(model_name):
    if model_name == MODEL_T5_AMR_PATH:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_BASE)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 't5' in model_name:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, default='t5-base',
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

    model_path = [i for i in model_names if args.model_path in i][0]
    
    #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
    from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration
    from lrml_train_pred import transform_lists_to_ds, train

    if 't5' in model_path:
        lrs = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4]
    else:
        lrs = [4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]

    bss = [4, 8, 12, 16]
    beam_nums = [1, 3, 5]
    for seed in [42]:
        for lr in lrs:
            for bs in bss:
                for beam_num in beam_nums:
                    exp_name = 'b' + str(bs) + '_' + str(lr) + '_beam' + str(beam_num) + '_' + args.name + '_' + str(seed)

                    hyperparameters = dict(epochs=args.epochs, learning_rate=lr, batch_size=bs, adafactor=False,
                                           decay=1.000e-4, warmup_steps=100, dataset=dataset_name, model_name=args.model_path,
                                           model_path=model_path, run_name=exp_name, eval_strategy='epoch',
                                           eval_steps=args.steps, early_stopping_patience=5, max_length=512,
                                           seed=seed, write_predictions=args.write_predictions, evaluate_only=args.evaluate_only,
                                           evaluation_set=args.eval, num_beams=beam_num
                                           )
                    print(hyperparameters)
                    tokenizer, model = get_tokenizer_and_model(model_path)
                    ds = get_datasets(tokenizer)
                    try:
                        train(datasets=ds, hyperparameters=hyperparameters, model=model,
                              tokenizer=tokenizer)
                    except Exception as e:
                        print(e)
