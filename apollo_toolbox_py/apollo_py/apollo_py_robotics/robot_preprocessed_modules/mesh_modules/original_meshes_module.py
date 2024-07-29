from typing import List, Optional, Dict
from apollo_rust_file_pyo3 import PathBufPy


class ApolloOriginalMeshesModule:
    def __init__(self, link_mesh_relative_paths: List[Optional[str]]):
        self.link_mesh_relative_paths: List[Optional[PathBufPy]] = list(
            map(lambda x: None if x is None else PathBufPy().append(x), link_mesh_relative_paths))

    def __repr__(self):
        return f"ApolloOriginalMeshesModule(link_mesh_relative_paths={list(map(lambda x: None if x is None else x.to_string(), self.link_mesh_relative_paths))})"

    @classmethod
    def from_dict(cls, data: Dict) -> 'ApolloOriginalMeshesModule':
        return cls(data['link_mesh_relative_paths'])
