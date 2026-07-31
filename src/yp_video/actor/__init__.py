"""Which visible person performed the annotated action.

Detection (``yp_video.person``) answers "who is on this frame"; this package
answers "which of them made the contact at (frame, x, y)". That is a separate
question from "who is this player" (``yp_video.reid``), it has its own human
labels and its own dataset — so it owns them here rather than borrowing the
ReID package's. The learned models live in yp-spot and are reached through
``spot_associate``; nothing in this package trains or stores one.

Layout, lowest first:
    labels           the durable human verdict + where it lives on disk
    resolution       how one extraction record's actor was resolved
    ranking          geometry policy: candidate generation and the rule decisions
    track_features   the numeric contract the track dataset is built on
    service          the single entry point extraction depends on
    dataset / evaluate   the corpus over human labels and how policies score on it

Nothing in here imports ``yp_video.reid``.
"""
