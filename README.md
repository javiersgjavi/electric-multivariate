# Multivariate Time Series Forecasting with Deep Learning


# Table of Contents
1. [Abstract](#Abstract) 
2. [Authors](#Authors) 
3. [How to execute the code](#Execution) 
4. [License](#license) 

<a name="Abstract"/>

## Abstract

An Extensive Comparative between Univariate and Multivariate Deep Learning Models in Day-Ahead Electricity Price Forecasting

> Electricity is a product that has greatly changed the way we think about the world. In fact, it is one of the indicators that show the progress of a civilisation. With the deregulation of the electricity market in the 1990s, electricity price forecasting became a fundamental task in all countries. This task was a challenge for both producers and consumers, as many factors had to be taken into account. In this paper, given the large number of factors that influence the electricity market price we have conducted several experiments to see how the use of multiple variables can improve the effectiveness of price prediction. Tests have been performed with different time periods using data from Spain and different Deep Learning models, showing that the use of these variables represents an improvement with respect to the univariate model, in which only the price was considered.

<a name="Authors"/>

## Authors:
- Belén Vega-Márquez
- Javier Solís-García
- Isabel A. Nepomuceno Chamorro
- Cristina Rubio-Escudero

<a name="Execution"/>

## How to execute the code

1. Modify required parameters in docker-compose.yaml

2. Build Docker image: ```docker build . -t electric```

3. Start Docker container: ```docker-compose up -d```

4. Check container id: ```docker ps```

5. Enter Docker container: ```docker exec -it <container_id> bash```

6. Execute ```./get_data.sh``` to obtain the data

7. Move to experiments folder: ```cd experiments/```

8. Give execution permissions to run.sh: ```chmod +x run.sh```

9. Execute: ```./run.sh``` to launch an experiment

10. The log of the experiment can be watched with the command ```tail -f experiment{GPU}.out``` where {GPU} is set in the file run.sh

**In addition, below is a video tutorial with an example of execution, which can be consulted in case you have any doubts.**

[![Alt text](https://img.youtube.com/vi/s4FNWdRoNhQ/0.jpg)](https://www.youtube.com/watch?v=s4FNWdRoNhQ)




## License<a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
