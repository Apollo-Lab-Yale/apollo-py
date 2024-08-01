from apollo_toolbox_py.apollo_py_blender.lines import ApolloBlenderLineSet

# b = ApolloBlenderLine.spawn_new([0., 0., 0.], [1., 1., 1.])
# print(b.line_mesh)

s = ApolloBlenderLineSet(100)
s.set_line_at_frame([0,0,0], [1,0,0], 0)
s.set_line_at_frame([0,0,0], [1,0,0], 1)
