**Challenge**: From protein language to features without alignments
====

An objective in computational biology is to be able to read and generate biology in its natural language. Transformer models are being trained to learn the language of proteins directly from protein sequences into a deep representation as embeddings. These language models should potentially be able to extract the same information content as evolutionary profiles.

A protein sequence dataset has been augmented by replacing the alignment profiles with embeddings. The embeddings are from a recent protein language model ESM-1b, that has been created and pretrained by Facebooks AI and research team. Your task is to predict secondary structure features from this dataset.

.. contents:: Table of Contents
   :depth: 2

Folder Structure
================

::

  ProteinLanguageChallenge/
  │
  ├── challenge/
  │    │
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── base/ - abstract base classes
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │
  │    ├── models/ - models, losses, and metrics
  │    │
  │    ├── trainer/ - training of the model
  │    │
  │    └── utils/ - logging, visualisation of tensorboard 
  │
  ├── logging.yml - logging configuration
  │
  ├── data/ - directory for storing input data
  │
  ├── experiments/ - directory for storing configuration files
  │
  └── saved/ - directory for checkpoints and logs

In this challenge you should work on the models and the data_loader folder. Everything else should already be ready to quickly start training your custom models.

Challenge aim
================
The challenge is simply to achieve the highest Q8/Q3 accuracy of secondary structure prediction on the CASP12 test dataset.

Requirements
=====
- Python 3.8 (I suggest to use either `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ or `pyenv <https://github.com/pyenv/pyenv>`_)
- `Docker <https://www.docker.com/>`_


Setup
=====

1. Download this repository
2. Create a environment from the root folder of the repository

.. code-block::

  $ conda env create --file environment.yml
  $ conda activate challenge

If you dont use conda you can use pip with an environment

.. code-block::

  $ pip install -r requirements.txt

3. Run the script that automatically downloads the datasets

.. code-block::

  $ sh get_data.sh

4. Set the challenge package a development

.. code-block::

  $ cd challenge
  $ python setup.py develop

This creates a symbolic link for the challenge.

Usage
=====

To train your model you can run the train command linking to your experiment config

.. code-block::

  $ challenge train -c experiments/config.yml

Example of a config file format. The config shows the default baseline model.
------------------
Config files are in `.yml` format:

.. code-block:: HTML

    name: baseline
    save_dir: saved/
    seed: 1234
    target_devices: [0]
    
    arch:
      type: Baseline
      args:
        in_features: 1280
    
    data_loader:
      type: ChallengeDataLoader
      args:
        train_path: [data/Train_ESM1b.npz]
        test_path: [data/CASP12_ESM1b.npz]
        dataset_loader: ChallengeDataOnlyEmbedding
        batch_size: 15
        nworkers: 2
        shuffle: true
        validation_split: 0.05
    
    loss: secondary_structure_loss
    
    metrics:
      metric_q8: 0
      metric_q3: 1
    
    optimizer:
      type: Adam
      args:
        lr: 0.0001
        weight_decay: 0
    
    training:
      early_stop: 3
      epochs: 50
      monitor: min val_loss
      save_period: 1
      tensorboard: true
    
    lr_scheduler:
      type: null
    
    augmentation:
      type: null


Add addional configurations if you need.

Using config files
------------------
Modify the configurations or create new `.yml` config files, then run:

.. code-block::

  $ challenge train -c experiments/config.yml

Evaluating models
------------------
Usually the models are evaluated after the training finishes. If you now want to check your pretrained model then you can run this. It will evaluate the the model with the test set in the experiment config.

.. code-block::

  $ challenge eval -c experiments/config.yml -m saved/path/to/model_best.pth

Resuming from checkpoints
-------------------------
You can resume from a previously saved checkpoint by:

.. code-block::

  challenge train -c experiments/config.yml -r path/to/checkpoint

Checkpoints
-----------
You can specify the name of the training session in config files:

.. code-block:: HTML

  "name": "Baseline"

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in
mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:

.. code-block:: python

  checkpoint = {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }

Tensorboard Visualization
--------------------------
This template supports `<https://pytorch.org/docs/stable/tensorboard.html>`_ visualization.

1. Run training

    Set `tensorboard` option in config file true.

2. Open tensorboard server

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at
    `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of
model parameters will be logged. If you need more visualizations, use `add_scalar('tag', data)`,
`add_image('tag', image)`, etc in the `trainer._train_epoch` method. `add_something()` methods in
this template are basically wrappers for those of `tensorboard.SummaryWriter` module.

**Note**: You don't have to specify current steps, since `TensorboardWriter` class defined at
`logger/visualization.py` will track current steps.

Output
================
It is important that your model returns the same size of out features as the baseline models forward method. You can see the forward method at ProteinLanguageChallenge/challenge/challenge/models/baseline/model.py

Benchmarking System
================
The continuous integration script in .github/workflows/ci.yml will automatically build the Dockerfile on every commit to the main branch. This docker image will be published as your hackathon submission to https://biolib.com/<YourTeam>/<TeamName>. For this to work, make sure you set the BIOLIB_TOKEN and BIOLIB_PROJECT_URI accordingly as repository secrets.