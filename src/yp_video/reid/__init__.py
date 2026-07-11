"""Player re-identification on annotated action frames.

Tracking-free by design: each action event already gives (frame, xy), so the
pipeline is detect persons on that single frame → pick the box the contact
point belongs to → crop → appearance embedding. Matching embeddings to player
identities is a separate, later concern.
"""
