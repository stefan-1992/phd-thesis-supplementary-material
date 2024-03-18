import os
import shutil

from datasets import Dataset
from transformers import T5ForConditionalGeneration, EarlyStoppingCallback, \
    T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
import evaluate
import random

from lrml_score import compute_lrml

default_tokenizer = T5Tokenizer.from_pretrained('t5-base')

metric = evaluate.load('bleu')


def transform_lists_to_ds(inputs, labels, sample=False, max_length=1024, tokenizer=None, padding='longest'):
    if sample:
        sample_size = int(len(inputs) / 10)
        inputs = random.sample(inputs, sample_size)
        labels = random.sample(labels, sample_size)
    if tokenizer is None:
        tokenizer = default_tokenizer
    input_tokens = tokenizer(inputs, padding=padding, truncation=True, max_length=max_length)
    output_tokens = tokenizer(labels, padding=padding, truncation=True, max_length=max_length).input_ids

    return Dataset.from_dict({'input_sample': inputs, 'input_ids': input_tokens.input_ids,
                              'attention_mask': input_tokens.attention_mask, 'labels': output_tokens,
                              'output_sample': labels
                              })


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    tok_preds = [default_tokenizer.decode(i, skip_special_tokens=True) for i in logits]
    tok_labels = [default_tokenizer.decode(i, skip_special_tokens=True) for i in labels]

    lrml_metric = compute_lrml(predictions=tok_preds, references=tok_labels, entity_weight=2, filter_empty=True)
    if sum([len(i) for i in tok_preds]) > 0: 
        lrml_metric.update(metric.compute(predictions=tok_preds, references=tok_labels))
    lrml_metric['prediction'] = tok_preds[0]
    wandb.log({'prediction': tok_preds[0]})
    return lrml_metric


def train(datasets, hyperparameters, model=None, tokenizer=None, delete_model=True):
    global default_tokenizer

    with wandb.init(project='Thesis', entity='stefan_fuchs_phd', config=hyperparameters):
        if hyperparameters:
            wandb.log(hyperparameters)
        config = wandb.config
        run_name = config['model_name'] + '-' + config['dataset'] + '-' + config['run_name']

        wandb.run.name = run_name

        training_args = Seq2SeqTrainingArguments('models/' + wandb.run.name,
                                                 evaluation_strategy=config['eval_strategy'],
                                                 save_strategy=config['eval_strategy'],
                                                 eval_steps=config['eval_steps'],
                                                 save_steps=config['eval_steps'],
                                                 warmup_steps=config['warmup_steps'],
                                                 adafactor=config['adafactor'],
                                                 metric_for_best_model='lrml_f_score',
                                                 eval_accumulation_steps=32,
                                                 num_train_epochs=config['epochs'],
                                                 learning_rate=config['learning_rate'],
                                                 per_device_train_batch_size=config['batch_size'],
                                                 seed=config['seed'],
                                                 per_device_eval_batch_size=config['batch_size'],
                                                 save_total_limit=2, load_best_model_at_end=True,
                                                 log_level='info',
                                                 report_to=['wandb'], weight_decay=config['decay'],
                                                 predict_with_generate=True,
                                                 generation_num_beams=config['num_beams'],
                                                 generation_max_length=config['max_length'])

        if model is None:
            model = T5ForConditionalGeneration.from_pretrained(config['model_path'], ignore_mismatched_sizes=True)
        if tokenizer is not None:
            default_tokenizer = tokenizer

        wandb.run.define_metric('eval/bleu', summary='max')
        wandb.run.define_metric('eval/score', summary='max')
        wandb.run.define_metric('eval/lrml_f_score', summary='max')

        valid_set_name = 'valid' if 'valid' in datasets.keys() else 'test'

        trainer = Seq2SeqTrainer(
            model=model, args=training_args,
            train_dataset=datasets['train'], eval_dataset=datasets[valid_set_name]
,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])]
        )

        if 'input_sample' in datasets[valid_set_name].column_names:
            wandb.log({'input_sample': datasets[valid_set_name]['input_sample'][0],
                       'output_sample': datasets[valid_set_name]['output_sample'][0]})

        if not config['evaluate_only']:
            # if 'checkpoint' in config['model_path']:
            #     trainer.train(config['model_path'])
            # else:
            trainer.train()

        if config['write_predictions']:
            os.makedirs('predictions', exist_ok=True)
            predictions, label_ids, metrics = trainer.predict(datasets['test'], metric_key_prefix='test')
            results = []
            scores = []
            for i in range(len(predictions)):
                res = compute_metrics(([predictions[i]], [label_ids[i]]))
                scores.append(res['lrml_f_score'])
                results.append(str(res))
            
            results.append('')
            results.append('')
            
            results.append('Max: ' + str(max(scores)) + '; Index: ' + str(scores.index(max(scores))) + '; Min: ' 
                           + str(min(scores)) + '; Index: ' + str(scores.index(min(scores))))
            results.append('')

            results.append('Max Text: ' + datasets['test']['input_sample'][scores.index(max(scores))])
            results.append('Max Label: ' + datasets['test']['output_sample'][scores.index(max(scores))])
            results.append(results[scores.index(max(scores))])
            results.append('Min Text: ' + datasets['test']['input_sample'][scores.index(min(scores))])
            results.append('Min Label: ' + datasets['test']['output_sample'][scores.index(min(scores))])
            results.append(results[scores.index(min(scores))])
            
                
            # predicted_sentences = [default_tokenizer.decode(i, skip_special_tokens=True) for i in predictions]
            with open('predictions/' + run_name + '.txt', 'w') as file:
                file.writelines('\n'.join(results) + '\n')

        trainer.evaluate(datasets[config['evaluation_set']], metric_key_prefix='test')

        if delete_model:
            shutil.rmtree('models/' + wandb.run.name)

        return 'models/' + wandb.run.name
