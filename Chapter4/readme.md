# Setting up your environment

Use Python Version 3.10.x

### Install required libraries using pip:
pip install -r requirements.txt

### Download the T5-AMR model and update the path in cib_hyper_training.py.
https://github.com/bjascob/amrlib-models  -> Select "parse_t5" and download the model

### Run the training script
python3 cib_hyper_training.py

### See possible configuration options
python3 cib_hyper_training.py --help

