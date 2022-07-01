FROM tensorflow/tensorflow:2.6.1-gpu
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN apt-get install -y wget
RUN python -m pip install --upgrade pip & pip install -r 'requirements.txt'
