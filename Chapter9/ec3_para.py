# Follows the translation setup https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd

from lrml_utils import *


dataset_name = 'lrml'

RANDOM_SPLIT = 'random_split'


df = pd.read_csv('data/lrml_ds_v8.csv')


def get_dataset(tokenizer, inputs, labels):
    train_ds = transform_lists_to_ds(
        tokenizer, inputs, labels)
    return train_ds


def get_splits(tokenizer, df, split_type, simplifications):
    completion_prefix = 'translate English to LegalRuleML: '
    # Train on maximum number of samples for production model
    train_index = df.loc[df[split_type] != 3].index
    valid_index = df.loc[df[split_type] == 3].index
    text, lrml = get_text_and_lrml(df, [])

    # LONGER INPUT
    train_completion_samples = [[(completion_prefix + text_ + '<sep>' + i, j) for i, j in get_auto_completion_training_samples2(
        lrml_, 2)] for text_, lrml_ in zip(text[train_index], lrml[train_index])]
    train_completion_samples_flat = [item for sublist in train_completion_samples for item in sublist] + [
        (completion_prefix + text_, lrml_) for text_, lrml_ in zip(text[train_index], lrml[train_index])]
    train_auto_completion_ds = get_dataset(tokenizer, [i[0] for i in train_completion_samples_flat], [
                                           i[1] for i in train_completion_samples_flat])
    text, lrml = get_text_and_lrml(df, ['pred'])
    valid_completion_samples_flat = [(completion_prefix + text_, lrml_) for text_, lrml_ in zip(text[valid_index], lrml[valid_index])]
    valid_auto_completion_ds = get_dataset(tokenizer, [i[0] for i in valid_completion_samples_flat], [
                                           i[1] for i in valid_completion_samples_flat])

    return {'train': train_auto_completion_ds, 'valid': valid_auto_completion_ds, 'test': valid_auto_completion_ds}


def node_to_lrml2(node, stop_node=None, separator=','):
    initial_depth = node.depth
    last_depth = -1
    lrml = ''
    for i in PreOrderIter(node):
        if i.depth > last_depth:
            if last_depth != -1:
                lrml += '('
        else:
            last_depth - i.depth
            lrml += ')' * (last_depth - i.depth)
            lrml += separator
        if stop_node is not None and i.node_id == stop_node.node_id:
            weights = [0.1]*len(i.name)
            for j in range(1, min(4, len(i.name))):
                weights[j] *= 2
            lrml += i.name[:random.choices(range(len(i.name)),
                                           weights, k=1)[0]]
            break
        lrml += i.name
        last_depth = i.depth
    #   Only add brackets for full print
    if stop_node is None:
        lrml += ')' * (last_depth - initial_depth)
    #   Remove root node
    if node.is_root:
        lrml = lrml.replace('root(', '')
        if stop_node is None:
            lrml = lrml[:-1]
    return lrml


def get_auto_completion_training_samples2(lrml, number):
    tree = parse_to_tree(lrml)
    return [get_auto_completion_pair2(tree) for i in range(number)]


def get_auto_completion_pair2(tree):
    choice = random.choice(tree.descendants)
    return node_to_lrml2(tree, stop_node=choice), node_to_lrml2(choice)


def get_file_names_from_df(df):
    return df['file'].str.split('-').apply(lambda x: x[1]).str.split('#').apply(lambda x: x[0])


def get_text_and_lrml(df, mode):
    lrml = df['label']
    if len(mode) == 0:
        text = pd.concat((df['input_original'], df['input_para']))
        lrml = pd.concat((df['label'], df['label']))
    elif 'pred' in mode:
        text = df['input_original']
    elif 'para' in mode:
        text = df['input_para']
    return text, lrml


def get_original(lrml, simplifications):
    return lrml


# We evaluate still on the original LRML
def clean_pred(lrml, simplifications, added_spaces=True):
    prefix = ' ' if added_spaces else ''

    # postprocessing_
    lrml = lrml.strip()
    lrml = lrml.replace('[', '(').replace(']', ')').replace(
        '{', '(').replace('}', ')')
    lrml = lrml.replace(').', ')')
    lrml = fix_then(lrml, prefix=prefix)
    if '(' in lrml:
        # Fix errors is postprocessing
        lrml = reverse_loop(lrml, prefix=prefix)
        lrml = reverse_resolve_expressions(lrml, fix_errors=True, prefix=prefix)
        lrml = reverse_combine_rel_and_var(lrml, prefix=prefix)
        lrml = reverse_move_and_or_to_data_node(lrml)
        lrml = reverse_units(lrml, prefix=prefix)

        # postprocessing
        lrml = remove_duplicate_expressions(lrml, prefix + 'obligation')
        lrml = remove_duplicate_expressions(lrml, prefix + 'expression')
    return lrml


def custom_postprocess_text(preds, labels, simplifications):
    preds = [clean_pred(pred, simplifications) for pred in preds]
    labels = [[clean_pred(label, simplifications)] for label in labels]

    return preds, labels


def fix_then(lrml, prefix):
    tree = parse_to_tree(lrml)
    if len(tree.children) == 1:
        thens = findall(tree, filter_=lambda x: ((x.name == prefix + 'then')))
        if len(thens) > 0:
            thens[0].parent = tree
    return node_to_lrml(tree)


def save_split_camel(text):
    regex = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
    regex2 = re.compile(r'(?<=[a-z])\.(?=[a-z])')
    if not ' ' in text:
        text = regex.sub(' ', text).lower().replace('_', ' ')
        text = regex2.sub('. ', text).replace('_', ' ')
    else:
        text = text.replace('_', ' ')
    return text


EXPERIMENT_NAMES = ['1to2_fullonly']


if __name__ == '__main__':
    parser = ArgumentParser(description="Trainer script",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mp', '--model-path', type=str, default='models/model_t5-amr/',
                        help='Model path to be used.')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--write-predictions', action='store_true')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='The learning rate.')
    parser.add_argument('-bs', '--batchsize', type=int, default=8,
                        help='The batch size used for training. Evaluation uses batchsize * 2.')
    parser.add_argument('-nb', '--num-beams', type=int, default=3,
                        help='Beam size for evaluation')
    parser.add_argument('--seed', type=str, default='43,44,45',
                        help='The random seed. Comma separated list allowed - 43,44,45. Default: 43')
    parser.add_argument('--steps', type=int, default=500,
                        help='Evaluate after how many steps?')
    parser.add_argument('--eval', type=str, default='test',
                        help='On which dataset should be evaluated after training?')
    parser.add_argument('--exp', type=str, required=True,
                        help='The experiment number.')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Which gpu should it run on')
    parser.add_argument('--epochs', type=int, default=20,
                        help='The random seed.')
    parser.add_argument('-esp', '--early-stopping-patience', type=int, default=20,
                        help='The early stopping patience.')
    parser.add_argument('-es', '--eval_strategy', type=str, default='epoch',
                        help='Strategy for evaluation: epoch or steps?')
    parser.add_argument('--skip', type=int, default=0,
                        help='Strategy for evaluation: epoch or steps?')

    args, unknown = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lr = args.learning_rate
    bs = args.batchsize


    for exp in args.exp.split(','):
        exp = int(exp)
        no_repeat_ngram_size = 9
        max_length = 512
        #     Late import to be able to set CUDA_VISIBLE_DEVICES before torch is initialised
        from egice_training import transform_lists_to_ds, train, init_tokenizer

        model_name = 't5-amr' if 't5-amr' in args.model_path else args.model_path
        tokenizer_name = 't5-base' if 't5-amr' in args.model_path else args.model_path

        tokenizer = init_tokenizer(tokenizer_name)

        gen_kwargs0 = {
            "max_length": max_length,
            "num_beams": args.num_beams,
            "early_stopping": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        gen_kwargss = [gen_kwargs0]

        skipped = 0

        for split in [RANDOM_SPLIT]:
            for seed in [43]:
                if skipped < args.skip:
                    skipped += 1
                    print('Skipping', skipped)
                    continue

                exp_name = model_name + '_' + \
                    str(exp) + '_' + \
                    EXPERIMENT_NAMES[exp] + '_' + split + '_esp' + str(
                        args.early_stopping_patience) + '_e' + str(args.epochs) + '_' + str(seed)

                df['input_original'] = 'translate English to LegalRuleML: ' + get_file_names_from_df(
                    df) + ' ' + df['text']
                df['input_para'] = 'translate English to LegalRuleML: ' + get_file_names_from_df(
                    df) + ' ' + df['paraphrase']
                df['label'] = df['lrml'].apply(
                    lambda x: get_original(x, [])).apply(tree_based_spacing)
                metric_for_best_model = 'lrml_f_score'

                def post_process(x, y): return custom_postprocess_text(
                    x, y, simplifications=[])

                ds = get_splits(tokenizer, df, split, [])
                hyperparameters = dict(epochs=args.epochs, lr=lr, bs=bs, adafactor=False, weight_decay=1.000e-1, metric_for_best_model=metric_for_best_model,
                                       num_warmup_steps=0, dataset=dataset_name, model_name=model_name, model_path=[args.model_path],
                                       post_process=post_process, lr_scheduler_type='linear', gradient_accumulation_steps=1,
                                       run_name=exp_name, eval_strategy=args.eval_strategy, eval_steps=args.steps,
                                       early_stopping_threshold=args.early_stopping_patience, max_length=max_length,
                                       seed=seed, write_predictions=args.write_predictions,
                                       evaluate_only=args.evaluate_only, evaluation_set=args.eval, runs=1, start_epoch=0, skip_evals=0,
                                       teacher_forcing_percentage=0.0, label_augmentation=0.0, retrain=True, mask_percentage=0.0, is_ir=False,
                                       fp16=False, split=split, experiment=EXPERIMENT_NAMES[exp], exp_num=exp)

                hyperparameters.update(
                    {k: v for k, v in gen_kwargs0.items()})

                print(hyperparameters)

                last_model_path, all_predictions, train_predictions = train(datasets=ds, hyperparameters=hyperparameters,
                                                                            delete_model=False, gen_kwargss=gen_kwargss, post_process=post_process)
