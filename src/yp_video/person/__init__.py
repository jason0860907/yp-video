"""People on a frame — detection and segmentation primitives.

The bottom layer of the vision stack: boxes, 17-point COCO skeletons and
instance masks, and nothing about what those people are FOR. Deciding which
of them performed an action lives in ``yp_video.actor``; deciding who they
are lives in ``yp_video.reid``. Neither direction is imported here, and that
is the point — this package answers "who is visible", a question that has
one right answer regardless of the consumer.
"""
