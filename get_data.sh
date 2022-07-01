#!/bin/bash
mkdir data/quarantine
mkdir data/fraud
mkdir data/normal

wget https://www.dropbox.com/s/t857z4a1ofuucde/train.csv?dl=1
wget https://www.dropbox.com/s/xjj1955hyuad9so/test.csv?dl=1
mv train.csv?dl=1 data/quarantine/train.csv
mv test.csv?dl=1 data/quarantine/test.csv

wget https://www.dropbox.com/s/np4hg8v7bykacco/train.csv?dl=1
wget https://www.dropbox.com/s/ekj675082j8od7j/test.csv?dl=1
mv train.csv?dl=1 data/fraud/train.csv
mv test.csv?dl=1 data/fraud/test.csv

wget https://www.dropbox.com/s/ndae94ophgvsn1r/train.csv?dl=1
wget https://www.dropbox.com/s/u74lia1zaa6oice/test.csv?dl=1
mv train.csv?dl=1 data/normal/train.csv
mv test.csv?dl=1 data/normal/test.csv

cd experiments

python generate_data.py

cd ..

mv data/quarantine data/_quarantine/
mv data/fraud data/_fraud/
mv data/normal data/_normal/


