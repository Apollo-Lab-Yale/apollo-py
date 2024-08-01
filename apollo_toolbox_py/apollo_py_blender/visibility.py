from easybpy.easybpy import unhide_in_viewport, unhide_in_render, hide_in_viewport, hide_in_render
import bpy

__all__ = ['set_visibility']


def set_visibility(object: bpy.types.Object, visible):
    if visible:
        unhide_in_viewport(object)
        unhide_in_render(object)
    else:
        hide_in_viewport(object)
        hide_in_render(object)
