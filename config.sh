# storage directories
export STORE_DIR=/data/dust/user/wiedersb/HH_DNN
export CACHE_DIR=${STORE_DIR}/cache
export PICTURE_DIR=${STORE_DIR}/pictures/features
export MODELS_DIR=${STORE_DIR}/models # saved directories
export TENSORBOARD_DIR=${STORE_DIR}/tensorboard # tensorboard storage

# debug level: DEBUG; INFO; WARNING
export LOG_LEVEL="DEBUG"
export FILE_LOG_LEVEL="DEBUG"

# location where the input data can be found.
export INPUT_DATA_DIR="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/prod20"
#export INPUT_DATA_DIR="/data/dust/user/riegerma/hh2bbtautau/run3_training_data/prod14"

# pyenv handling: not necessary when own environment is used
export ML_ENV="ml_pytorch" # venv name
export PYENV_ROOT="/afs/desy.de/user/w/wiedersb/.pyenv/" # pyenv root
