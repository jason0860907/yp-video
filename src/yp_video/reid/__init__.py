"""Which player is this — appearance embeddings over the extracted crops.

Tracking-free by design: extraction already reduced each action event to one
actor crop (see yp_video/extraction), so this package never opens a video to
find a person. It embeds those crops, clusters and matches them against the
names a human assigned, exports datasets for yp-reid to train on, and reads
the resulting checkpoints back.

Layout, lowest first:
    store         embedding matrices and the players annotation path
    embedder      crop paths → L2-normalized vectors (spawns yp-reid)
    identity      clustering, assignments and nearest-centroid matching
    sessions      which videos share a player name-space
    dataset / evaluate / checkpoints / metrics   the learning loop

Which person performed the action a crop was cut from is a different
question with different labels — ``yp_video.actor`` owns it, and nothing in
here imports it. Following a person through a rally is a third
(``yp_video.tracklets``), which both of them read.
"""
