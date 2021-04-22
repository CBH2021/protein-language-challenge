FROM python:3.8-slim AS predict

# move final model
# COPY <path to model> model.pth
COPY saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth model.pth

# move final configuration
# RUN mv <path to config> config.yml
RUN mv experiments/config.yml config.yml

# install dependencies
COPY challenge/requirements.txt challenge/requirements.txt
RUN pip install -r challenge/requirements.txt

# copy challenge project
COPY challenge challenge
COPY README.rst README.rst

# install challenge as package
RUN pip install -e challenge