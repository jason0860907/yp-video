"""Which visible person performed the annotated action.

Detection (``yp_video.person``) answers "who is on this frame"; this package
answers "which of them made the contact". That is a separate question from
"who is this player" (``yp_video.reid``), with its own human labels — so it
owns them here rather than borrowing the ReID package's. The model is yp-spot's
joint person/action head, reached through ``person_action``.

Layout, lowest first:
    labels           the durable human verdict + where it lives on disk
    resolution       how one extraction record's actor was resolved
    policy           what reassociation asks a policy and what it gets back
    candidates       tracklets near an event, as the head's candidate boxes
    person_action    the joint person/action head across the SPOT boundary
    review           review progress over the human verdicts

Nothing in here imports ``yp_video.reid``.
"""
