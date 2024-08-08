from easybpy.easybpy import add_keyframe
import bpy

__all__ = ['KeyframeUtils']


class KeyframeUtils:
    @staticmethod
    def _keyframe_abstract(object: bpy.types.Object, channel_str: str, frame: int):
        add_keyframe(object, channel_str, frame)

    @staticmethod
    def keyframe_visibility(object: bpy.types.Object, frame: int):
        KeyframeUtils._keyframe_abstract(object, 'hide_viewport', frame)
        KeyframeUtils._keyframe_abstract(object, 'hide_render', frame)

    @staticmethod
    def keyframe_rotation(object: bpy.types.Object, frame: int):
        KeyframeUtils._keyframe_abstract(object, 'rotation_euler', frame)

    @staticmethod
    def keyframe_location(object: bpy.types.Object, frame: int):
        KeyframeUtils._keyframe_abstract(object, 'location', frame)

    @staticmethod
    def keyframe_scale(object: bpy.types.Object, frame: int):
        KeyframeUtils._keyframe_abstract(object, 'scale', frame)

    @staticmethod
    def keyframe_transform(object: bpy.types.Object, frame: int):
        KeyframeUtils.keyframe_location(object, frame)
        KeyframeUtils.keyframe_rotation(object, frame)
        KeyframeUtils.keyframe_scale(object, frame)
