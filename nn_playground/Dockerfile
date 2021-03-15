FROM continuumio/miniconda3

RUN conda install pytorch
RUN conda install scikit-learn
RUN conda install pandas
RUN conda install matplotlib
RUN conda install tensorboard

ADD train.py train.py
ADD dataset.py dataset.py
ADD data_coniguration.json data_coniguration.json
ADD train_coniguration.json train_coniguration.json
ADD visualize_utils.py visualize_utils.py
RUN mkdir output
RUN mkdir logs