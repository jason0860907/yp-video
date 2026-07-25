"""People followed through a rally — who is on court, over time.

Detection (``yp_video.person``) answers "who is visible on this frame". This
package answers "and which of them is the same person one frame later",
producing one tracklet per person per rally with its boxes and instance masks.

That makes it the natural unit for both downstream questions: which tracklet
performed an action (``yp_video.actor``) and who that tracklet is
(``yp_video.reid``). Both read tracklets; neither is imported here, and
tracking deliberately does not read the action annotation either — it needs
to know where the rallies are, not what happened inside them.

    store       where tracklets and their packed masks live
    geometry    resolving a box back to the tracklet it belongs to
    tracking    the dense per-rally detect + ByteTrack pass
"""
