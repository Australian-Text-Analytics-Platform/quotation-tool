from abc import ABCMeta, abstractmethod
import pathlib
from typing import Iterable, Callable, Generator
import pandas as pd
import os
import codecs
from chardet.universaldetector import UniversalDetector


class FlaggedPath(object):
    def __init__(self, path: pathlib.Path, reason: str):
        self.path = path
        self.reason = reason

    def __repr__(self):
        return f"Path: {self.path}\t [REASON: {self.reason}]"


class Check(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, path):
        raise NotImplementedError()


class FileSizeCheck(Check):
    REASON: str = "Exceeded the maximum size of {} bytes."

    def __init__(self, max_bytes: int):
        self.max_size = max_bytes

    def __call__(self, path: pathlib.Path) -> bool:
        # print(f"File: {path} Size: {path.stat().st_size} bytes")
        return path.stat().st_size < self.max_size

    def reason(self):
        return self.REASON.format(self.max_size)


def check_file_lang(file: pathlib.Path):
    # print(f"file_lang: checking file - {file}")
    return True


class EncodingCheck(Check):
    """
    This Encoding Check depends on well known chardet.

    There seems to be better libraries on Github.
    TODO: https://github.com/Ousret/charset_normalizer
    """

    REASON: str = "File encoding inferred to be {} with confidence {}. Expected {}."

    expected_but_got_still_fine = {
        'UTF-8': {'ASCII'}
    }

    def __init__(self, expected: str, min_rows_to_check: int):
        try:
            codecs.lookup(expected)
        except LookupError:
            raise LookupError(f"Encoding {expected} is not supported.")
        self._expected = expected.upper()
        self._min_rows_to_check = min_rows_to_check

        self._detector = UniversalDetector()

        self._current_inferred = ''
        self._current_inferred_conf = -1

    @property
    def expected(self):
        return self._expected

    def __call__(self, path: pathlib.Path) -> bool:
        detector = self._detector
        with open(path, 'rb') as h:
            for i, line in enumerate(h):
                detector.feed(line)
                if detector.done and i > self._min_rows_to_check:
                    break
        detector.close()
        self._current_inferred = detector.result.get('encoding')
        self._current_inferred_conf = detector.result.get('confidence')
        if self.expected.upper() == self._current_inferred.upper():
            return True
        else:
            return self._current_inferred.upper() in self.expected_but_got_still_fine.get(self.expected.upper())

    def reason(self):
        return self.REASON.format(self._current_inferred, self._current_inferred_conf, self.expected)


class FileCheckers(object):
    def __init__(self, checks: list[Callable]):
        self._checks = checks
        self._flagged = dict()
        self._passed = list()

    def run(self, paths: Iterable[pathlib.Path]):
        if isinstance(paths, Generator):
            paths = list(paths)
        paths = self._to_paths(paths)
        flagged = dict()
        path: pathlib.Path
        for path in paths:
            for check in self.checks:
                passed = check(path)
                if not passed:
                    if isinstance(check, Check):
                        reason = check.reason()
                    else:
                        reason = check.__name__

                    key: str = str(path)
                    reasons = flagged.get(key, list())
                    reasons.append(reason)
                    flagged[key] = reasons
        self._passed = [str(p) for p in paths if str(p) not in flagged.keys()]
        self._flagged = flagged
        return flagged

    @property
    def checks(self):
        return self._checks

    def flagged(self):
        return self._flagged

    def passed(self):
        return self._passed

    def summary(self):
        num_flagged = len(self._flagged.keys())
        num_passed = len(self._passed)
        return pd.Series([num_flagged, num_passed, num_flagged + num_passed],
                         index=['Flagged', 'Passed', 'Total'],
                         name='File Check Summary',
                         dtype='UInt16')  # up to ~65k files

    def _to_paths(self, paths):
        for i, p in enumerate(paths):
            if not isinstance(p, pathlib.Path):
                paths[i] = pathlib.Path(p)
        return paths


if __name__ == '__main__':
    checks = [
        check_file_lang,
        FileSizeCheck(max_bytes=1_000_000)  # 1Mb
    ]
    file_checks = FileCheckers(checks)

    HOME = os.getenv("HOME")
    paths = pathlib.Path(f"{HOME}/Downloads/Data/Top100_Text_1/").rglob("*.txt")

    flagged = file_checks.run(paths)

    print("+++ Results +++")
    print("FLAGGED ", file_checks.flagged())
    print("PASSED ", file_checks.passed())
    print(file_checks.summary())
