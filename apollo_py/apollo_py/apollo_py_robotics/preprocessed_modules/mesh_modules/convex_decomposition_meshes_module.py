from typing import List, Dict
from apollo_rust_file_pyo3 import PathBufPy


class ApolloConvexDecompositionMeshesModule:
    def __init__(self, stl_link_mesh_relative_paths: List[List[str]], obj_link_mesh_relative_paths: List[List[str]], glb_link_mesh_relative_paths: List[List[str]]):
        self.stl_link_mesh_relative_paths: List[List[PathBufPy]] = list(
            map(lambda x: list(map(lambda y: PathBufPy().append(y), x)), stl_link_mesh_relative_paths))
        self.obj_link_mesh_relative_paths: List[List[PathBufPy]] = list(
            map(lambda x: list(map(lambda y: PathBufPy().append(y), x)), obj_link_mesh_relative_paths))
        self.glb_link_mesh_relative_paths: List[List[PathBufPy]] = list(
            map(lambda x: list(map(lambda y: PathBufPy().append(y), x)), glb_link_mesh_relative_paths))

        self._stl_link_mesh_relative_paths = stl_link_mesh_relative_paths
        self._obj_link_mesh_relative_paths = obj_link_mesh_relative_paths
        self._glb_link_mesh_relative_paths = glb_link_mesh_relative_paths

    def __repr__(self):
        return (f"ApolloConvexDecompositionMeshesModule("
                f"stl_link_mesh_relative_paths={self._stl_link_mesh_relative_paths}, "
                f"obj_link_mesh_relative_paths={self._obj_link_mesh_relative_paths},"
                f"glb_link_mesh_relative_paths={self._glb_link_mesh_relative_paths} "
                f")")

    @classmethod
    def from_dict(cls, data: Dict) -> 'ApolloConvexDecompositionMeshesModule':
        return cls(data['stl_link_mesh_relative_paths'], data['obj_link_mesh_relative_paths'], data['glb_link_mesh_relative_paths'])