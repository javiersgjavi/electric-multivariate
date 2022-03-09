#!/bin/bash
GPU=0
LOG_FILE=./experiments${GPU}.out
DATASETS=('../data/_*')
MODELS=('lstm' 'cnn' 'tcn' 'mlp' 'gru')
MODELS_ML=('tree' 'rf')
PARAMETERS=./parameters.json
OUTPUT=../results
CSV_FILENAME=results.csv

python main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --models_ml ${MODELS_ML[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
