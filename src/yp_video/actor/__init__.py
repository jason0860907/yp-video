"""Which visible person performed the annotated action.

Detection (``yp_video.person``) answers "who is on this frame"; this package
answers "which of them made the contact at (frame, x, y)". That is a separate
question from "who is this player" (``yp_video.reid``), it has its own human
labels, its own dataset, its own model and its own checkpoints — so it owns
them here rather than borrowing the ReID package's.

Layout, lowest first:
    labels        the durable human verdict + where it lives on disk
    resolution    how one extraction record's actor was resolved
    ranking       geometry policy: candidate generation and the rule decisions
    features      the numeric contract shared by training and inference
    model         the serializable learned ranker
    checkpoints   candidate storage and explicit shadow activation
    service       the single entry point extraction depends on
    dataset / train / evaluate   the learning loop over human labels

Nothing in here imports ``yp_video.reid``.
"""
