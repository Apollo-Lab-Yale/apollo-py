from apollo_toolbox_py.prelude import *
from apollo_toolbox_py.prelude_bpy import *

r = ResourcesRootDirectory.new_from_default_apollo_robots_dir()
s = r.get_subdirectory('b1z1')
c = s.to_chain_tensorly()
spawn = False

start_state = [-0.05739262351617041, -0.14797593706584578, 0.047515595680537454, -0.03844258278497295, -0.04722270332258458, -0.14686619314391988, 0.1501421531885781, -0.08955473711738092, 0.09066687869281101, 0.041949251559591, -0.08576421618308211, 0.1265370531942326, -0.017102623813716546, -0.19950901552762823, -0.060064520885626, -0.10347451826958372, 0.024262757379761447, 0.09789642294267616, -0.1245192109029659, -0.02801018420469701, -0.05688881323090485, -0.06047569245386528, -0.08219134673487992, 0.0594926237131896]
end_state = [-0.01087281676362566, -0.0010568737301828923, -0.0045810186573736885, 0.09452271116445472, -0.00784419349618377, 0.7173793244271015, 0.043592623024045776, -0.06412533002909944, 0.09307830835744473, 0.02656538858157745, -0.07109946794083946, 0.12143463535007949, 0.025638180651027867, -0.0733960734460554, 0.0029882889168710534, 0.007304189041966489, -0.08125086852041256, 0.042186103618374174, -0.014097745297741047, 0.4361355078995244, -0.5012193273669763, 0.049725290253267176, 0.02254440386116827, 0.01964920663913214]

if spawn:
    cc1 = ChainBlender.spawn(c, r)
    cc2 = ChainBlender.spawn(c, r)
else:
    cc1 = ChainBlender.capture_already_existing_chain(0, c, r)
    cc2 = ChainBlender.capture_already_existing_chain(1, c, r)

cc1.set_state(start_state)
cc2.set_state(end_state)

cc1.set_all_links_plain_mesh_alpha(0.01)

l = BlenderLineSet(1000)
l.set_line_at_frame([0., 0., 0.], [0., 1., 0.], 1, color=(1.0, 1.0, 0.0, 1.0))
