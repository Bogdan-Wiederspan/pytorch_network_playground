# storage directories
export STORE_DIR=/data/dust/user/wiedersb/HH_DNN # ROOT of storage
export CACHE_DIR=${STORE_DIR}/cache # directory where preprocessed data is stored
export PICTURE_DIR=${STORE_DIR}/pictures/features
export MODELS_DIR=${STORE_DIR}/models # saved models
export TENSORBOARD_DIR=${STORE_DIR}/tensorboard # tensorboard storage

# debug level: DEBUG; INFO; WARNING
export LOG_LEVEL="DEBUG"
export FILE_LOG_LEVEL="DEBUG"

# location where the input data can be found.
export INPUT_DATA_DIR="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/prod20"
#export INPUT_DATA_DIR="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/prod14"

# virtualenv handling
export VENV_MODE="pyenv" # pyenv or cf, handles switch
export ML_ENV="ml_pytorch" # venv name of virtualenv
export CF_SANDBOX="venv_hbt_dev" #  sandbox name in columnflow
export PYENV_ROOT="/afs/desy.de/user/w/wiedersb/.pyenv/" # root of pyenv installation
