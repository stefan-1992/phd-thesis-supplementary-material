import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd


dataset_name = 'lrml'

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


def term_if_true(bool_val, term):
    if bool_val:
        return term
    return ''


def contains_any(comments, terms):
    if pd.isna(comments):
        return False
    return len([i for i in terms if i in comments]) > 0


def is_tacit_added(row):
    old = row['text_old']
    new = str(row.get('text', ''))
    start_index = new.find(old.replace(';', '')[:20])
    return 'TACIT-' in str(row.get('comments','')) or start_index > 2


def get_datasets():
    prefix = 'parse English to LegalRuleML: '
    datasets = {}

    tags = {'ignore': ['Abstract', 'CausesNotTranslated', 'ContextIgnore', 'FilteringRequiredByModel', 'Ignore',
                       'LearnToIgnore', 'LearnToIgnoreFigure', 'ModelIgnoreExplanation', 'NotEncodedParts',
                       'OrOtherOnly', 'ReferenceNotTranslated', 'ReferenceOverDescription', 'RemovableContext',
                       'SomeDetailsNotEncoded', 'UselessInfoInBrackets'],
            'fine': ['Detailed', 'DetailedParts', 'FineGrained', 'AndOrDetailed', 'CoarseThenFine'],
            'tacit': ['DomainKnowledge', 'MissingInformation', 'Tacit', 'Tacit - Formula', 'TacitReference',
                      'UnpredictableParts', 'ImplicitPrecondition'],
            'coarse': ['LongData', 'LongEntities', 'SplitableEntities'],
            'medium': ['MediumData', 'OneSplitableEntity', 'PartlyAbstract', 'SomeLongEntities', 'CoarseThenFine']}

    bc = lrml_df_v7['file'].str.replace('lrml/NZ_NZBC-', '').str.lower().str.split('#').apply(
        lambda x: x[0])

    med_ = lrml_df_v7['comments'].apply(lambda x: contains_any(x, tags['medium']))
    coarse_ = lrml_df_v7['comments'].apply(lambda x: contains_any(x, tags['coarse']))
    fine_ = ~(med_ | coarse_)
    medium = med_.apply(lambda x: term_if_true(x, 'medium-grained'))
    coarse = coarse_.apply(lambda x: term_if_true(x, 'coarse-grained'))
    fine = fine_.apply(lambda x: term_if_true(x, 'fine-grained'))
    med_coars_fine = medium + coarse + fine

    define = lrml_df_v7['lrml'].str.contains('function( define)', regex=False).apply(
        lambda x: term_if_true(x, 'define'))  # 69
    rulestatement = lrml_df_v7['lrml'].str.contains('rulestatement(', regex=False).apply(
        lambda x: term_if_true(x, 'rulestatement'))  # 48
    key = lrml_df_v7['lrml'].str.contains('relation( key', regex=False).apply(lambda x: term_if_true(x, 'key'))

    tacit_ = lrml_df_v7['comments'].apply(lambda x: contains_any(x, tags['tacit']))
    tacit_added_ = lrml_df_v7.apply(is_tacit_added, axis=1)
    domain_ = lrml_df_v7['comments'].apply(lambda x: contains_any(x, tags['domain']))
    tacit_filtered_ = (tacit_ & ~tacit_added_) | domain_
    tacit_filtered = tacit_filtered_.apply(lambda x: term_if_true(x, 'implicit'))

    ignore = lrml_df_v7['comments'].apply(lambda x: contains_any(x, tags['ignore'])).apply(
        lambda x: term_if_true(x, 'ignore'))
 

    text, lrml = lrml_df_v7['text'], lrml_df_v7['lrml']

    datasets['condition4_end_tok_lrml_all_bc'] = get_doc_split(lrml_df_v7, prefix + text + '</s>' + bc, lrml)
    datasets['condition4_end_tok_lrml_all_define'] = get_doc_split(lrml_df_v7, prefix + text + '</s>' + define, lrml)
    datasets['condition4_end_tok_lrml_all_rulestatement'] = get_doc_split(lrml_df_v7,
                                                                          prefix + text + '</s>' + rulestatement, lrml)
    datasets['condition4_end_tok_lrml_all_key'] = get_doc_split(lrml_df_v7, prefix + text + '</s>' + key, lrml)
    datasets['condition4_end_tok_lrml_all_med_coars_fine'] = get_doc_split(lrml_df_v7,
                                                                           prefix + text + '</s>' + med_coars_fine,
                                                                           lrml)
    datasets['condition4_end_tok_lrml_all_tacit_filtered'] = get_doc_split(lrml_df_v7,
                                                                           prefix + text + '</s>' + tacit_filtered,
                                                                           lrml)
    datasets['condition4_end_tok_lrml_all_ignore'] = get_doc_split(lrml_df_v7, prefix + text + '</s>' + ignore, lrml)
    datasets['condition4_end_tok_lrml_all_med'] = get_doc_split(lrml_df_v7, prefix + text + '</s>' + medium, lrml)
    datasets['condition4_end_tok_lrml_all_coarse'] = get_doc_split(lrml_df_v7, prefix + text + '</s>' + coarse, lrml)

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
