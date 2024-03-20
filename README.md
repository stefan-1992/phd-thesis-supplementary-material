# Supplementary Material for the PhD Thesis with Title:
Transformer-based Semantic Parsing of Building Regulations: Supporting Regulators in Drafting Machine-Interpretable Rules 
## Author
Stefan Martin Fuchs

## Chapter 3: Natural Language Processing for Building Code Interpretation (Fuchs and Amor, 2021)
The supplementary material for the systematic literature review is available in the folder `Chapter3`. The folder contains the following files:
- InitialLiteratureReview.csv: Literature retrieved from the initial search in Google Scholar, with screening and full-text assessment results.
- KeywordClustering.svg: Visual clustering of keywords extracted from the literature review.
- KeywordClusteringOverview.svg: Overview of the most important keywords and clusters.


## Chapter 4: Transformer-based Semantic Parsing of Building Regulations (Fuchs et al., 2022)
The supplementary material for the transformer-based semantic parsing of building regulations is available in the folder `Chapter4`. The folder contains the following files: 
- data/lrml_ds_v1.csv: LRML Dataset
- data/cib_results.csv: Results for the hyperparameter search and model comparison

- lrml_score.py: Functions to calculate the LRML F1 score
- lrml_train_pred.py: Functions for training and predicting LRML
- cib_hyper_training.py: Execute the hyperparameter search and model comparison
- cib_reporting.ipynb: Plotting the results
- cib_training_thesis.py: Training the transformer-based semantic parser baseline without pretraining


## Chapter 5: Improving Semantic Parsing through Additional Training Data (Fuchs et al., 2022)
The supplementary material for "Improving Semantic Parsing through Additional Training Data" is available in the folder `Chapter5`. The folder contains the following files: 
- data/lrml_ds_v1.csv: LRML Dataset
- data/silver_data.csv: Silver dataset
- data/cib_silver_results.csv: Results for the data augmentation experiments
- data/cib_multi_results.csv: Results for the multi-task learning experiments
- data/cib_multi_predictions_for_analysis: Predictions for qualitative analysis

- lrml_score.py: Functions to calculate the LRML F1 score
- lrml_train_pred.py: Functions for training and predicting LRML
- cib_silver_training.py: Training with silver data; Options: Train with silver dataset once, or multiply it for a higher weighting. 
- cib_silver_gold_training.py: Training with silver and gold data.
- cib_silver_refine_training.py: Refine the model trained on silver data with gold data only.
- cib_multi_training.py: Multi-task training with all specified datasets. TODO: Add scripts for retrieval of multi-task training datasets.
- cib_silver_gold_training.py: Multi-task training with semantic parsing and regulation data only.
- cib_silver_refine_training.py: Refine the model trained using multi-task training with gold data only.
- cib_prediction.py: Prediction only - Write results to file.

## Chapter 6: Consistent Formal Representation of Building Regulations (Fuchs et al., 2023c)
- data/lrml_ds_v1.csv: Original LRML Dataset
- data/lrml_ds_v2.csv: LRML Dataset with invalid LRML rules removed
- data/lrml_ds_v3.csv: LRML Dataset with improved alignments - Step 1 - With LRML tokenisation improved
- data/lrml_ds_v4.csv: LRML Dataset with tacit knowledge added to the input clauses - Step 2 - With LRML tokenisation improved
- data/lrml_ds_v5.csv: LRML Dataset with LRML rules cleansed - Steps 3-5 - With LRML tokenisation improved
- data/lrml_ds_v6.csv: LRML Dataset with manual reference cleansing - Base for Reference and Entity cleaning - Without LRML tokenisation improved
- data/lrml_ds_v7.csv: LRML Dataset with all rules and all data cleansing steps applied - With LRML tokenisation improved
- data/lrml_ds_v8.csv: LRML Dataset with evaluation-based cleansing applied - Without LRML tokenisation improved - With Reversible IR applied (was reverted for this chapter)
- data/lrml_ds_doc-exp.csv: Baseline dataset for document-based evaluation - With LRML tokenisation improved
- data/ecppm_results.xlsx: Results for experiments in Chapter 7. The three runs per setup were averaged.
- data/ecppm_thesis_results.csv: Results for evaluation-based cleansing. The three runs per setup were averaged.
- data/eval_pred_annotated_errors.csv: Analysis of Chapter5/data/cib_multi_predictions_for_analysis to identify error classes and their implications

- lrml_score.py: Functions to calculate the LRML F1 score
- lrml_train_pred.py: Functions for training and predicting LRML
- lrml_utils.py: Utility functions for the LRML representation
- lrml_baseline_short_training.py: Experiment to establish initial baseline with 512 token length
- lrml_baseline_training.py: Experiment to establish initial baseline with all other interventions
- lrml_cleansing_training.py: Experiments to step-wise evaluate the cleansing process
- lrml_conditioning_training.py: Experiments to evaluate the value conditioning
- lrml_remove_training.py: Evaluate the removal of clauses in the value conditioning experiment
- lrml_document_training.py: Experiments to evaluate the document-based evaluation
- lrml_random_training.py: Evaluate the random training in the document-based evaluation experiment
- ecppm_ir_lrml_baseline_training.py: Experiment on the document based-split after the final evaluation-based cleansing
- ecppm_ir_lrml_random_training.py: Experiment on the random split after the final evaluation-based cleansing


## Chapter 7: Improving the Semantic Parsing of Building Regulations through Intermediate Representations (Fuchs et al., 2023b)
- data/results/*: Results for the experiments in Chapter 7. The three runs per setup were averaged in the _combined files.

- egice_results.ipynb: Generate the _combined result files.
- egice_training.py: Training the T5 model with a custom training loop.
- training_utils.py: Utility functions for the training loop.
- egice_experiment_rev_ir.py: Reversible IR experiments.
- egice_experiment_ir.py: Lossy IR experiments.
- egice_experiment_paraphrase.py: Paraphrase as IR experiments.
- egice_experiment_consistency.py: Self Consistency experiments.
- egice_experiment_paraphrase_noir.py: Training with paraphrases without IRs experiment.
- egice_experiment_thesis.py: Baseline without generation parameters.

## Chapter 8: Using Large Language Models for the Interpretation of Building Regulations (Fuchs et al., in press)
- eppm_preds/*: Predictions and csv files with the used used exemplars for all experiments in Chapter 8.
- prompts/*: Prompts for contextualisation and chain-of-thought prompting.
- data/eppm_num_samples.csv: How many samples where used in the GPT-4 per clause sampling experiments
- data/eppm_results.csv: The exeperimental results as logged with Weights and Biases.
- data/lrml_additional.csv: Intermediate results for the additional training data experiments.
- data/lrml_ds_v8_add_data_150.csv: The additional training data added to lrml_ds_v8_sel.
- data/lrml_ds_v8_gen_data.csv: The recreated dataset lrml_ds_v8_sel using GPT-4.
- data/lrml_ds_v8_sel.csv: The V8 LRML dataset with columns including the manual selection of the exemplar rules.
- eppm_experiment.ipynb: The main experiment file for the GPT-4 experiments.
- eppm_data_train.py: Training T5 with GPT4 as teacher model.
- egice_training.py/lrml_score.py/training_utils.py: The utility files for the training and evaluation of the T5 model as in Chapter 7.
- lrml_utils.py: Additional utility functions for the LRML reversible IR in Chapter 7.


## Chapter 9: Transformer-based Autocompletion for Semi-Automated Translation into Formal Representations (Fuchs et al., 2023a)
- To be added soon.

# Disclaimer
The legal clauses used to train and evaluate the transformer-based semantic parser come from the Acceptable Solutions and Verification Methods for the New Zealand Building Codes (https://www.building.govt.nz/building-code-compliance/). The legal clauses are used for research purposes only and are not intended to be used for any other purpose. The legal clauses are not up-to-date and should not be used for any regulatory or compliance purposes. For alignment purposes, some paragraphs were split into multiple clauses, and some information has been removed. The legal clauses are provided as is and without any warranty.

The LegalRuleML rules are based on Dimyadi et al. (2020) (https://github.com/CAS-HUB/nzbc-lrml). To establish a consistent semantic parsing dataset, some rules were modified, merged, or removed. Please report any errors in this repository. These errors will be fixed for the next version of the dataset. The LegalRuleML rules are provided as is and without any warranty.

# References

- Dimyadi, J., Fernando, S., Davies, K., & Amor, R. (2020). Computerising the New Zealand building code for automated compliance audit. New Zealand Built Environment Research Symposium (NZBERS).
- Fuchs, S. and Amor, R. (2021). Natural language processing for building code inter- pretation: A systematic literature review. In Proc. of the Conference CIB W78, volume 2021, pages 11–15.
- Fuchs, S., Witbrock, M., Dimyadi, J., and Amor, R. (2022). Neural semantic parsing of building regulations for compliance checking. In IOP Conference Series: Earth and Environmental Science, volume 1101.
- Fuchs, S., Dimyadi, J., Ronee, A. S., Gupta, R., Witbrock, M., and Amor, R. (2023a). A legalruleml editor with transformer-based autocompletion. In EC3 Conference 2023, volume 4, pages 0–0. European Council on Computing in Construction.
- Fuchs, S., Dimyadi, J., Witbrock, M., and Amor, R. (2023c).Training on digitised building regulations for automated rule extraction. In eWork and eBusiness in Architecture, Engineering and Construction: ECPPM 2022. CRC Press.
- Fuchs, S., Dimyadi, J., Witbrock, M., and Amor, R. (2023b). Improving the semantic parsing of building regulations through intermediate representations. In EG-ICE 2023 Workshop on Intelligent Computing in Engineering, Proceedings.
- Fuchs, S., Witbrock, M., Dimyadi, J., and Amor, R. (in press). Using large language models for the interpretation of building regulations. In 13th International Conference on Engineering, Project, and Production Management (EPPM2023).