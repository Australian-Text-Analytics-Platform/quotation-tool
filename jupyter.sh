#!/bin/zsh

# This script serves jupyter with the correct settings
# settings:
# + allow 3GB max buffer size
# + allow 2GB websocket window size   (allows for uploads < 2GB using fileupload ipywidget)
# + use Safari, firefox and chrome tested to crash with large file uploads despite increased window size.

PATH_CONFIG="./jupyter_notebook_config.py"
DIR_NOTEBOOKS="./"

if [[ ! -f $PATH_CONFIG ]]; then
  echo "[warn] $PATH_CONFIG not found. Expect some limitations. Read script documentation."
  PATH_CONFIG=""
fi

# debug purposes only
[[ $1 == "--debug" ]] && jupyter notebook --config "$PATH_CONFIG" --NotebookApp.show_config True && exit 0

jupyter lab \
  --config "$PATH_CONFIG" \
  --NotebookApp.browser='chrome' \
  $@ \
  $DIR_NOTEBOOKS
