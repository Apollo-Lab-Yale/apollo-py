from typing import List, Optional, Dict
from apollo_rust_file_pyo3 import PathBufPy


class ApolloConvexHullMeshesModule:
    def __init__(self, stl_link_mesh_relative_paths: List[Optional[str]],
                 obj_link_mesh_relative_paths: List[Optional[str]], glb_link_mesh_relative_paths: List[Optional[str]]):
        self.stl_link_mesh_relative_paths: List[Optional[PathBufPy]] = list(
            map(lambda x: None if x is None else PathBufPy().append(x), stl_link_mesh_relative_paths))

        self.obj_link_mesh_relative_paths: List[Optional[PathBufPy]] = list(
            map(lambda x: None if x is None else PathBufPy().append(x), obj_link_mesh_relative_paths))

        self.glb_link_mesh_relative_paths: List[Optional[PathBufPy]] = list(
            map(lambda x: None if x is None else PathBufPy().append(x), glb_link_mesh_relative_paths))

    def __repr__(self):
        return (f"ApolloConvexHullMeshesModule("
                f"stl_link_mesh_relative_paths={list(map(lambda x: None if x is None else x.to_string(), self.stl_link_mesh_relative_paths))}, "
                f"obj_link_mesh_relative_paths={list(map(lambda x: None if x is None else x.to_string(), self.obj_link_mesh_relative_paths))},"
                f"glb_link_mesh_relative_paths={list(map(lambda x: None if x is None else x.to_string(), self.glb_link_mesh_relative_paths))} "
                f")")

    @classmethod
    def from_dict(cls, data: Dict) -> 'ApolloConvexHullMeshesModule':
        return cls(data['stl_link_mesh_relative_paths'], data['obj_link_mesh_relative_paths'],
                   data['glb_link_mesh_relative_paths'])
