# Multivariate Time Series Forecasting with Deep Learning
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![tfa-nightly 0.11](https://img.shields.io/badge/TensorFlow%20Addons-0.11.0.dev20200601015706-FF6F00?logo=tensorflow)](https://github.com/tensorflow/addons/releases)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-380/)

An Extensive Comparative between Univariateand Multivariate Deep Learning Models inDay-Ahead Electricity Price Forecasting

> Electricity  is  a  product  that  has  greatly  changed  the  waywe think about the world, in fact it is one of the indicators that showthe  progress  of  a  civilisation.  With  the  deregulation  of  the  electricitymarket in the 1990s, electricity pricing became a fundamental task in allcountries. This task was a challenge for both producers and consumers,as many factors had to be taken into account. In this paper, given thelarge  number  of  factors  that  influence  the  electricity  market  price,  amethodology is proposed that makes use of exogenous variables to predictthe electricity market price. Tests have been carried out with differenttime periods for data from Spain and different Deep Learning models,showing that the use of these variables is an improvement with respectto the univariate model, in which only the price was used.

## Authors:
- Belén Vega-Márquez
- Javier Solís García
- Isabel A. Nepomuceno Chamorro
- Cristina Rubio Escudero

## Execution

1. Modify required parameters in docker-compose.yaml

2. Build Docker image: ```docker build . -t electric```

3. Start Docker container: ```docker-compose up -d```

4. Check container id: ```docker ps```

5. Enter Docker container: ```docker exec -it <container_id> bash```

6. Move to experiments folder: ```cd electric-multivariate/experiments/```

7. Give execution permissions to run.sh: ```chmod +x run.sh```

8. Execute: ```nohup ./run.sh```


## License<a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
