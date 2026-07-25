"""Turning annotated action events into per-event actor records.

The roof of the vision stack: ``pipeline`` runs person detection → actor
association → crop → embedding, and the two application services apply the
corrections a human makes on top of that output. This is the ONLY package
allowed to depend on ``person``, ``actor`` and ``reid`` at once — the three
of them never reach for each other.

``store`` is the exception that keeps the layering honest: it is a leaf
holding the on-disk shape of what extraction produces (the record jsonl and
the actor crops), so ``actor`` and ``reid`` can read that output without
depending on the pipeline that writes it.
"""
