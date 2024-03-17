import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

from datasets import Dataset, concatenate_datasets


dataset_name = 'multi-silver'

lrml_df = pd.read_csv('data/lrml_ds_v1.csv')



def get_splits(input_series, label_series, shuffle=False, test_size=0.1, max_length=512, tokenizer=None):
    dataset = transform_lists_to_ds(input_series.tolist(), label_series.tolist(), max_length=max_length,
                                    tokenizer=tokenizer, padding='max_length')
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def get_datasets():
    prefix = 'parse English to LegalRuleML: '

    text = lrml_df['text']
    lrml = lrml_df['lrml']
    base_original_512_ds = get_splits(prefix + text, lrml, shuffle=False, max_length=512)
    
    all_datasets_dict = {}
    path = 'datasets/multi_task_t5_base/'

    # old_dataset_names = ['record', 'lrml', 'rte', 'sst2', 'cola', 'boolq', 'scot', 'cb', 'wnli', 'stsb', 'mnli', 'amr',
    #                      'cogs', 'qnli', 'multirc', 'wic', 'copa', 'qqp', 'cfq', 'mrpc']
    old_dataset_names = ['lrml', 'scot', 'amr', 'cogs', 'cfq']
    for folder in os.listdir(path):
        if folder in old_dataset_names:
            all_datasets_dict[folder] = Dataset.load_from_disk(path + folder)
    
    # Limit the number of samples per dataset to the number of lrml samples
    all_datasets_dict['amr'] = Dataset.from_dict(all_datasets_dict['amr'][:21800])
    all_datasets_dict['cfq'] = Dataset.from_dict(all_datasets_dict['cfq'][:21800])
    all_datasets_dict['lrml-gold'] = concatenate_datasets([base_original_512_ds['train'] for i in range(20)])

    # all_datasets_dict['lrml-gold'] = concatenate_datasets([base_original_512_ds['train'] for i in range(20)])
    all_ds = concatenate_datasets([i for i in all_datasets_dict.values()])

    return {'train': all_ds, 'test': base_original_512_ds['test']}


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, default='t5-base',
                        help='Model path to be used.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--write-predictions', action='store_true')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=4e-4,
                        help='The learning rate.')
    parser.add_argument('-bs', '--batchsize', type=int, default=12,
                        help='The batch size used for training. Evaluation uses batchsize * 2.')
    parser.add_argument('--num-beams', type=int, default=3,
                        help='Beam size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='The random seed.')
    parser.add_argument('--steps', type=int, default=500,
                        help='Evaluate after how many steps?')
    parser.add_argument('--eval', type=str, default='test',
                        help='On which dataset should be evaluated after training?')
    parser.add_argument('--name', type=str, default='multi-sem-smaller',
                        help='The experiment name, what makes this run special?')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Which gpu should it run on')
    parser.add_argument('--epochs', type=int, default=2,
                        help='The random seed.')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lr = args.learning_rate
    bs = args.batchsize
   
    #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
    from lrml_train_pred import transform_lists_to_ds, train
    from transformers import T5ForConditionalGeneration

    for seed in [42]:
        exp_name = 'b' + str(bs) + '_' + str(lr) + '_beam' + str(args.num_beams) + '_' + args.name + '_' + str(seed)

        hyperparameters = dict(epochs=args.epochs, learning_rate=lr, batch_size=bs, adafactor=False,
                               decay=1.000e-4, warmup_steps=100, dataset=dataset_name, model_name='T5-base',
                               model_path=args.model_path, run_name=exp_name, eval_strategy='steps',
                               eval_steps=args.steps, early_stopping_patience=30, max_length=512,
                               seed=seed, write_predictions=args.write_predictions, evaluate_only=args.evaluate_only,
                               evaluation_set=args.eval, num_beams=args.num_beams
                               )
        print(hyperparameters)
        try:
            train(datasets=get_datasets(), hyperparameters=hyperparameters,
                  model=T5ForConditionalGeneration.from_pretrained(args.model_path),
                  delete_model=False)
        except Exception as e:
            print(e)
