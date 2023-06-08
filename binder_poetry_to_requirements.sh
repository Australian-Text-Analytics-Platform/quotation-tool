#!/bin/zsh
# export poetry dependencies to requirements.txt
# binderhub requires

echo "This script will export poetry dependencies into requirements.txt used in binder."
echo "A new temporary virtual environment will be created and subsequently deleted. "

VENV_DIR='./.venv-poetry-to-requirements'
POETRY_FILE="./pyproject.toml"
REQ_FILE='./requirements.txt'

if [[ ! -f $POETRY_FILE ]]; then
  echo "-- $POETRY_FILE not found. Exited."
  exit 1;
fi

if [[ -f $REQ_FILE ]]; then
  printf "-- $REQ_FILE exist. Replace? (y/n): "
  read x
  [[ $x != 'y' ]] && echo "Exited." && exit 0
  rm -rf $REQ_FILE
fi

if [[ -d $VENV_DIR ]]; then
  printf "-- Virtual environment $VENV_DIR already exists. Replace(y/n)? "
  read x
  [[ $x != 'y' ]] && echo "Exited." && exit 0
  rm -rf $VENV_DIR
fi

echo "++ Building new temporary virtual env at $VENV_DIR... "
python3 -m venv $VENV_DIR

echo "++ Activating virtual env..."
source $VENV_DIR/bin/activate
which python3

echo "++ Installing dependencies..."
pip install --upgrade pip
pip install poetry
poetry install --without dev # note no extras are installed. (e.g. apple dependencies.)

echo "++ Exporting poetry dependencies to $REQ_FILE..."
poetry export --without-hashes --without dev --format=requirements.txt > $REQ_FILE

echo "++ Cleaning up..."
rm -rf $VENV_DIR

echo "++ Done."
echo "++ To test on binder, please commit and push $REQ_FILE so binder docker container is updated."
exit 0
