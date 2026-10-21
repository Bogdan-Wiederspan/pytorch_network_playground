# pytorch_network_playground

PyTorch training framework for the HH → bbττ DNN. The python package lives in `src/bbTT`.

## 1. Get the code

Login to github, click _Fork_ and then _Create fork_. You now have a personal fork at
`github.com/YOUR-USERNAME/pytorch_network_playground`.

In your terminal go to the location where you want the code to live and write:
```bash
git clone git@github.com:YOUR-USERNAME/pytorch_network_playground.git
cd pytorch_network_playground
git remote add upstream git@github.com:Bogdan-Wiederspan/pytorch_network_playground.git
```

## 2. Create your config
To setup the project run.
```bash
source setup.sh
```
The first run creates `config.sh` from `config.sh.example` and stops.

Open `config.sh` and check:

| Variable | Meaning |
| --- | --- |
| `STORE_DIR` | Root for everything the framework writes (cache, models, tensorboard, evaluation, pictures). Default: `/data/dust/user/$USER/HH_DNN` |
| `CREATE_DIR` | `1` lets `setup.sh` create the directories under `STORE_DIR` (`mkdir -p`, safe to leave on), `0` skips it |
| `TRAINING_ROOT`, `ERA` | Where the preprocessed input data lives. Ask if you do not have access. |
| `VENV_ROOT`, `ML_ENV` | Directory containing the virtualenv and its name. Must match step 2 (`/data/dust/user/$USER/pyenv_virtualenvs` and `ml_torch`). |


## 3. Install the python environment (once)
Execute the shellscript by copy into the terminal
```
# uses python3,
bash install_venv.sh
# OR when a specfic version is used
bash install_venv.sh python3.11
```

Creates the virtualenv at `${VENV_ROOT}/${ML_ENV}` from your `config.sh` and installs `requirements.txt`
(`torch` with CUDA 12.1 wheels, plus this package in editable mode).

## 4. Every new shell

```bash
source setup.sh
```

This activates the virtualenv, creates missing storage directories and adds the CUDA libraries shipped with the
pip packages (cudnn, nccl) to `LD_LIBRARY_PATH`, so training works even if the system CUDA installation does not
match the torch wheel.

## What you can do

All scripts accept `--help`, which lists the available options.

**Train** a model. Hyperparameters live in `src/bbTT/configs/` (`debug_config.py` for quick tests,
`full_config.py` for the real thing). Models go to `MODELS_DIR`, logs to `TENSORBOARD_DIR`.

```bash
cd src/bbTT
python train/train.py --help
```

**Evaluate** a trained checkpoint on the test, training or validation set and save the scores to `EVALUATION_DIR`:

```bash
python evaluation/evaluate.py --help
```

**Export** a checkpoint to torch-export (`.pt2`) with a dynamic batch dimension:

```bash
python evaluation/export/pt2.py --help
```

**Monitor** the training with tensorboard:

```bash
tensorboard --logdir "${TENSORBOARD_DIR}"
```

**Lint** with the settings from `pyproject.toml`:

```bash
ruff check src && ruff format src
vulture        # dead code
```

## Debugging

`setup.sh` provides two shell functions that drop you into a debugger whenever a python script crashes with an
uncaught exception, at the line where it happened and with all local variables available:

```bash
pdb_on          # IPython shell (default), alternatives: pdb_on pdb / pdb_on ipdb
python train/train.py
pdb_off         # back to normal tracebacks
```

Nothing in the code needs to change; the hook works through `PYTHONPATH` and `.setup/sitecustomize.py`.

## Working with git

Push your changes to your fork:

```bash
git add ...
git commit ...
git push
```

Pull in changes from upstream (then `git push` to update your fork, or "sync" the fork on github.com):

```bash
git pull upstream main
```

Open a pull request to the original repo with the _Contribute_ button on github.com.
