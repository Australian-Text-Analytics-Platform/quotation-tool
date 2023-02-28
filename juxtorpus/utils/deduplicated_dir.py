from typing import Union
import pathlib, os, shutil, tempfile
import hashlib
import filecmp
import zipfile

import logging.config

logger = logging.getLogger(__name__)


class DeduplicatedDirectory(object):
    """
    The DeduplicatedDirectory is a proxy to a temporary directory that does not hold any duplicated files.

    It does this by keeping track of each file's message digest in an index.
    """

    def __init__(self, dir_path: Union[str, pathlib.Path] = None):
        if dir_path is None: dir_path = tempfile.mkdtemp()
        if isinstance(dir_path, str): dir_path = pathlib.Path(dir_path)
        if dir_path.exists() and not dir_path.is_dir():
            raise FileExistsError(f"{dir_path} exists and is not a directory. Please specify another path.")
        if not dir_path.exists(): dir_path.mkdir()
        self._dir_path = dir_path
        self._index = dict()
        self._hash_alg = hashlib.md5

    @property
    def path(self) -> pathlib.Path:
        return self._dir_path

    def files(self, hidden: bool = False) -> list[pathlib.Path]:
        """ Lists all the absolute paths of the files in the directory """
        f: pathlib.Path
        if hidden:
            cond = lambda f: f.is_file()
        else:
            cond = lambda f: f.is_file() and f.stem[0] != '.'
        return list(f for f in pathlib.Path(self._dir_path).glob('**/*') if cond(f))

    def list(self) -> list[str]:
        """ Lists all the name of the files in the directory. """
        return [f.name for f in self.files()]

    def add(self, file: pathlib.Path):
        """ Adds a file to the directory. Raises error if file already exists. """
        if not file.is_file():
            raise ValueError(f"{file.name} is not a file.")
        added = self._add_from(file, self._dir_path)
        self._build_index()
        return added

    def _add_from(self, file: pathlib.Path, start_at: pathlib.Path):
        if not start_at.exists():
            os.mkdir(start_at)
        if self.exists(file):
            raise ValueError(f"{file.name} already exists.")
        new_file_path = start_at.joinpath(file.name)
        shutil.copy(file, new_file_path)
        return new_file_path

    def add_content(self, content: bytes, fname: str):
        """ Add the content to the directory. Raises error if both file name and content are duplicated."""

        # note: to change this to content ONLY, remove the self._filename_exists condition and keep content_exists ONLY.
        if self._filename_exists(fname) and self.content_exists(content):
            raise ValueError(f"File name: {fname} and its content are duplicated.")
        new_file_path = self._dir_path.joinpath(fname)
        with open(new_file_path, 'wb') as fh: fh.write(content)
        self._build_index()
        return new_file_path

    def add_zip(self, path: pathlib.Path, verbose=False):
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{path} is not a zipfile.")
        with zipfile.ZipFile(path, 'r') as z:
            temp_dir = pathlib.Path(tempfile.mkdtemp())
            temp_dir = temp_dir.joinpath(pathlib.Path(z.filename).name)
            z.extractall(temp_dir)
            return self.add_directory(temp_dir, verbose)

    def add_directory(self, path: pathlib.Path, verbose=False):
        """ Adds a directory and its files. Raises error if directory name already exists. """
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")
        if self._filename_exists(path.name, root_only=True):
            raise ValueError(f"{path} already exists.")
        added = self._add_directory_from(path, self._dir_path, verbose=verbose)
        self._build_index()
        return added

    def _add_directory_from(self, path: pathlib.Path, start_at: pathlib.Path, verbose=False):
        if not start_at.is_dir():
            raise ValueError(f"{start_at} must be a directory.")
        added = []
        for file in path.glob('./*'):
            if file.is_file():
                try:
                    added.append(self._add_from(file, start_at))
                    if verbose: logger.info(f"{file.name} successfully added.")
                except ValueError:
                    if verbose: logger.warning(f"{file.name} already exist. Skipped.")
            elif file.is_dir():
                _next_dir = start_at.joinpath(file.name)
                if not _next_dir.exists(): os.mkdir(_next_dir)
                added += self._add_directory_from(file, _next_dir)
            else:
                logger.warning(f"{file} is neither a file or directory. Skipped.")
        return added

    def remove(self, fname: str):
        """ Removes the file from the directory. Raises error if file name did not match anything. """
        for existing in self.files():
            if existing.name == fname:
                os.remove(existing)
                for digest, path in self._index.items():
                    if path == existing:
                        del self._index[digest]
                        return
        raise ValueError(f"{fname} does not exist.")

    def exists(self, file: pathlib.Path, shallow: bool = True) -> bool:
        """ Check if file already exists in the directory.

        :param file: path to file.
        :param shallow: True - checks file metadata only. False - checks content.

        Uses filecmp under the hood.
        """
        for existing in self.files():
            if filecmp.cmp(file, existing, shallow=shallow):
                return True
        return False

    def content_exists(self, content: bytes, shallow: bool = True) -> bool:
        """ Check if content exists in directory.
        :param content:
        :param shallow: True - check size only. False - checks content.
        """
        size = len(content)
        digest = ''
        if not shallow:
            digest = self._hash_alg(content).hexdigest()
        for existing in self.files():
            if shallow:
                if existing.stat().st_size == size:
                    return True
            else:
                if self._index.get(digest, None) is not None:
                    return True
        return False

    def _filename_exists(self, fname: str, root_only=False):
        glob_pattern: str = '**/*'
        if root_only:
            glob_pattern = './*'
        for file in pathlib.Path(self._dir_path).glob(glob_pattern):
            if file.name == fname:
                return True
        return False

    def _build_index(self):  # todo: this should be async
        for existing in self.files():
            if existing in self._index.values():
                continue
            with open(existing, 'rb') as fh:
                digest = self._hash_alg(fh.read()).hexdigest()
            self._index[digest] = existing

    def _add_to_index(self, digest: str, path: pathlib.Path):
        self._index[digest] = path
