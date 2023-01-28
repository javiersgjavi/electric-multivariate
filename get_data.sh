#!/bin/bash
mkdir data/quarantine_exp1
mkdir data/fraud_exp1
mkdir data/normal_exp1
mkdir data/high_prices_exp1

mkdir data/quarantine_exp2
mkdir data/fraud_exp2
mkdir data/normal_exp2
mkdir data/high_prices_exp2

wget https://www.dropbox.com/s/f3ofvlkb0yysub1/train.csv?dl=0
wget https://www.dropbox.com/s/j4dwqubp9geapu2/test.csv?dl=0
mv train.csv?dl=0 data/quarantine_exp1/train.csv
mv test.csv?dl=0 data/quarantine_exp1/test.csv

wget https://www.dropbox.com/s/h7gw85jpdpj580l/train.csv?dl=0
wget https://www.dropbox.com/s/cr1n7gwe4i45jcq/test.csv?dl=0
mv train.csv?dl=0 data/fraud_exp1/train.csv
mv test.csv?dl=0 data/fraud_exp1/test.csv

wget https://www.dropbox.com/s/7tiqyxqwghrbdmz/train.csv?dl=0
wget https://www.dropbox.com/s/rj09hj61bwquu11/test.csv?dl=0
mv train.csv?dl=0 data/normal_exp1/train.csv
mv test.csv?dl=0 data/normal_exp1/test.csv

wget https://www.dropbox.com/s/zle187roxni3bmg/train.csv?dl=0
wget https://www.dropbox.com/s/z0fqe8q4qhdxc9r/test.csv?dl=0
mv train.csv?dl=0 data/high_prices_exp1/train.csv
mv test.csv?dl=0 data/high_prices_exp1/test.csv

wget https://www.dropbox.com/s/e182v1o1y6mem1d/train.csv?dl=0
wget https://www.dropbox.com/s/u0uy1dlgkwsgjpg/test.csv?dl=0
mv train.csv?dl=0 data/quarantine_exp2/train.csv
mv test.csv?dl=0 data/quarantine_exp2/test.csv

wget https://www.dropbox.com/s/7un9kxxj2bgb40b/train.csv?dl=0
wget https://www.dropbox.com/s/a0xhbl3242jhjgs/test.csv?dl=0
mv train.csv?dl=0 data/fraud_exp2/train.csv
mv test.csv?dl=0 data/fraud_exp2/test.csv

wget https://www.dropbox.com/s/ah1rteb88z3vpdi/train.csv?dl=0
wget https://www.dropbox.com/s/8hgje2dof78qt8q/test.csv?dl=0
mv train.csv?dl=0 data/normal_exp2/train.csv
mv test.csv?dl=0 data/normal_exp2/test.csv

wget https://www.dropbox.com/s/l9rfajcuvbloy3q/train.csv?dl=0
wget https://www.dropbox.com/s/vyn0i0m5qsi2cf1/test.csv?dl=0
mv train.csv?dl=0 data/high_prices_exp2/train.csv
mv test.csv?dl=0 data/high_prices_exp2/test.csv

cd experiments

python generate_data.py

cd ..

mv -T data/quarantine_exp1/ data/_quarantine_exp1/
mv -T data/fraud_exp1/ data/_fraud_exp1/
mv -T data/normal_exp1/ data/_normal_exp1/
mv -T data/high_prices_exp1/ data/_high_prices_exp1/

mv -T data/quarantine_exp2/ data/_quarantine_exp2/
mv -T data/fraud_exp2/ data/_fraud_exp2/
mv -T data/normal_exp2/ data/_normal_exp2/
mv -T data/high_prices_exp2/ data/_high_prices_exp2/

