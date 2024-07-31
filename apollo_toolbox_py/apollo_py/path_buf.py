from typing import List, Optional

from apollo_rust_file_pyo3 import PathBufPy
import copy


class PathBufPyWrapper:
    def __init__(self):
        self.path_buf = PathBufPy()

    @classmethod
    def _new_from_path_buf(cls, path_buf: PathBufPy) -> 'PathBufPyWrapper':
        out = cls.__new__(cls)
        out.path_buf = path_buf
        return out

    @classmethod
    def new_from_home_dir(cls) -> 'PathBufPyWrapper':
        return cls._new_from_path_buf(PathBufPy.new_from_home_dir())

    @classmethod
    def new_from_documents_dir(cls) -> 'PathBufPyWrapper':
        return cls._new_from_path_buf(PathBufPy.new_from_documents_dir())

    @classmethod
    def new_from_desktop_dir(cls) -> 'PathBufPyWrapper':
        return cls._new_from_path_buf(PathBufPy.new_from_desktop_dir())

    @classmethod
    def new_from_default_apollo_robots_dir(cls) -> 'PathBufPyWrapper':
        return cls._new_from_path_buf(PathBufPy.new_from_default_apollo_robots_dir())

    def append(self, s: str) -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.append(s))

    def append_vec(self, v: List[str]) -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.append_vec(v))

    def append_without_separator(self, s: str) -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.append_without_separator(s))

    def append_path(self, s: 'PathBufPyWrapper') -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.append_path(s.path_buf))

    def split_into_strings(self) -> 'List[str]':
        return self.path_buf.split_into_strings()

    def split_into_path_bufs(self) -> 'List[PathBufPyWrapper]':
        return [self._new_from_path_buf(v) for v in self.path_buf.split_into_path_bufs()]

    def walk_directory_and_find_first(self, s: 'PathBufPyWrapper') -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.walk_directory_and_find_first(s.path_buf))

    def walk_directory_and_find_all(self, s: 'PathBufPyWrapper') -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.walk_directory_and_find_all(s.path_buf))

    def create_directory(self):
        self.path_buf.create_directory()

    def delete_file(self):
        self.path_buf.delete_file()

    def delete_directory(self):
        self.path_buf.delete_directory()

    def delete_all_items_in_directory(self):
        self.path_buf.delete_all_items_in_directory()

    def copy_file_to_destination_file_path(self, s: 'PathBufPyWrapper'):
        self.path_buf.copy_file_to_destination_file_path(s.path_buf)

    def extra_last_n_segments(self, n: int) -> 'PathBufPyWrapper':
        return self._new_from_path_buf(self.path_buf.extract_last_n_segments(n))

    def get_all_items_in_directory(self, include_directories: bool = True, include_hidden_directories: bool = True, include_files: bool = True, include_hidden_files: bool = True) -> 'List[PathBufPyWrapper]':
        res = self.path_buf.get_all_items_in_directory(include_directories, include_hidden_directories, include_files,
                                                       include_hidden_files)
        return [self._new_from_path_buf(p) for p in res]

    def get_all_filenames_in_directory(self, include_hidden_files: bool = True) -> List[str]:
        return self.path_buf.get_all_filenames_in_directory(include_hidden_files)

    def read_file_contents_to_string(self) -> Optional[str]:
        return self.path_buf.read_file_contents_to_string()

    def write_string_to_file(self, s: str):
        self.path_buf.write_string_to_file(s)

    def __repr__(self):
        return f"PathBufPy({self.path_buf.to_string()})"

    def __str__(self):
        return f"PathBufPy({self.path_buf.to_string()})"
