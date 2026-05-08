# storage directories
export STORE_DIR=/data/dust/user/wiedersb/HH_DNN # ROOT of storage
export CACHE_DIR=${STORE_DIR}/cache # directory where preprocessed data is stored as well as dataset paths
export PICTURE_DIR=${STORE_DIR}/pictures/features
export MODELS_DIR=${STORE_DIR}/models # saved models
export TENSORBOARD_DIR=${STORE_DIR}/tensorboard # tensorboard storage

# debug level: DEBUG; INFO; WARNING
export LOG_LEVEL="DEBUG"
export FILE_LOG_LEVEL="DEBUG"

# location where the input data can be found.
export TRAINIG_ROOT="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/"
export ERA=prod20 # possible eras: prod14, prod20, prod24 (20 only has 22pre)
export INPUT_DATA_DIR="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/${ERA}"

# virtualenv handling
export VENV_MODE="venv" # pyenv, venv or cf, handles switch
export ML_ENV="ml_torch" # venv name of virtualenv
# export ML_ENV="ML_CF" # venv name of virtualenv
export VENV_ROOT="/data/dust/user/wiedersb/pyenv_virtualenvs/"
export CF_SANDBOX="venv_hbt_dev" #  sandbox name in columnflow
export PYENV_ROOT="/afs/desy.de/user/w/wiedersb/.pyenv/" # root of pyenv installation
