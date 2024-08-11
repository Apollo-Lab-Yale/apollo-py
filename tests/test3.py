from easybpy.easybpy import location, get_object

from apollo_toolbox_py.prelude import *

c = BlenderCube.spawn_new([0, 0, 0], [0, 0, 0], [1, 1, 1])
location(c.blender_object, [1, 2, 3])
print(c.name)


