# Supplementary Material for the PhD Thesis
## Title 
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
- To be added soon.

## Chapter 7: Improving the Semantic Parsing of Building Regulations through Intermediate Representations (Fuchs et al., 2023b)
- To be added soon.

## Chapter 8: Using Large Language Models for the Interpretation of Building Regulations (Fuchs et al., in press)
- To be added soon.

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