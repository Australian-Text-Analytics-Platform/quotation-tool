import tempfile
from typing import Callable
from ipywidgets import FileUpload, Output, VBox, widgets
from IPython.display import display
import pathlib
import logging.config

logger = logging.getLogger(__name__)

from juxtorpus.viz import Widget
from juxtorpus.utils import DeduplicatedDirectory

"""
NOTE: File size limit using jupyter notebooks.

start jupyter notebook with:
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
"""


class FileUploadWidget(Widget):
    DESCRIPTION = "Upload your files here.\n({})"
    ERR_FAILED_UPLOAD = "File upload unsuccessful. Please try again!."

    default_accepted_extensions = ['.csv', '.zip']

    def __init__(self, dir_path: pathlib.Path = None, accept_extensions: list[str] = None):
        if accept_extensions is None:
            accept_extensions = self.default_accepted_extensions

        self._dir = DeduplicatedDirectory(dir_path)

        self._uploader = FileUpload(
            description=self.DESCRIPTION.format(' '.join(accept_extensions)),
            accept=', '.join(accept_extensions),
            multiple=True,  # True to accept multiple files
            error=self.ERR_FAILED_UPLOAD,
            layout=widgets.Layout(width='320px')
        )
        # required for ipython renders
        self._output = Output()
        self._uploader.observe(self._on_upload, names=['value'])  # 'value' for when any file is uploaded.

        self._callback = None

    def uploaded(self, hidden: bool = False):
        return self._dir.files(hidden)

    def widget(self):
        return display(VBox([self._uploader, self._output]))

    def set_callback(self, callback: Callable):
        if not callable(callback): raise ValueError(f"Callback must be a function or a callable.")
        self._callback = callback

    def _on_upload(self, change):
        with self._output:
            new_files = self._get_files_data(change)
            for fdata in new_files:
                content = fdata.get('content')
                fname = fdata.get('name')
                if fname.endswith('.zip'):
                    added = self._add_zip(content, fname)
                else:
                    added = self._add_file(content, fname)
                if callable(self._callback): self._callback(self, added)

    def _get_files_data(self, change):
        new = change.get('new')
        if isinstance(new, dict):  # support v7.x
            fdata_list = list()
            for fname, fdata in new.items():
                fdata['name'] = fname
                fdata_list.append(fdata)
            return fdata_list
        return new  # support v8.x

    def _add_zip(self, content, fname):
        try:
            logger.info(f"Extracting {fname} to {self._dir.path}. Please wait...")
            tmp_zip_dir = pathlib.Path(tempfile.mkdtemp())
            tmp_zip_file = tmp_zip_dir.joinpath(fname)
            with open(tmp_zip_file, 'wb') as fh:
                fh.write(content)
            added = self._dir.add_zip(tmp_zip_file, verbose=True)
            hidden = [f for f in added if f.stem[0] == '.']
            logger.info(f"Done. Extracted {len(added)} files. {len(hidden)} are hidden.")
            return added
        except Exception as e:
            logger.info(f"Failed. Reason: {e}")
            return []

    def _add_file(self, content, fname):
        try:
            logger.info(f"Writing {fname} to {self._dir.path}...")
            added = self._dir.add_content(content, fname)
            logger.info("Success.")
            return added
        except Exception as e:
            print(f"Failed. Reason: {e}")
