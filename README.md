# pytorch_network_playground

## Fork repository

Login to github and click _Fork_ and then _Create fork._

Now, you have a personal fork at `github.com/YOUR-USERNAME/pytorch_network_playground`.

## Clone your fork

```bash
git clone git@github.com:YOUR-USERNAME/pytorch_network_playground.git
cd pytorch_network_playground
```

## Add original repo as upstream

```bash
git remote add upstream git@github.com:Bogdan-Wiederspan/pytorch_network_playground.git
```

## Adapt config.sh

Change _USER_ and _DNN_ROOT_ variables.

## Install python environment (only once)

```bash
mkdir /data/dust/user/${USER}/pyenv_virtualenvs
python -m venv /data/dust/user/${USER}/pyenv_virtualenvs/ml_torch
source /data/dust/user/${USER}/pyenv_virtualenvs/ml_torch/bin/activate
pip install -r requirements.txt
```

## Setup training environment (everytime you open a new shell)

```bash
source setup.sh
```

## Now you can

### ...start a training

```bash
cd src/bbTT
python train/train.py
```

### ...make changes to your own fork

```bash
git add ...
git commit ...
git push
```

### ...pull in changes from upstream

```bash
git pull upstream main
```

(This changes your local working copy, so you still need to _push_ to update your remote fork. Alternatively, you can also "sync" your fork on github.com and then pull in the changes.)

### ...open a pull request to the original repo

Use the _Contribute_ button on github.com.
