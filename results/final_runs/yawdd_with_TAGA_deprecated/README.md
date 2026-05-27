# Deprecated YawDD TAGA-era runs

This directory archives historical YawDD artifacts generated before Stage 9.

These runs used a `PCH + TAGA` style architecture: `feature_layout=sequence`, `temporal_head=transformer`, and an implicit trainable TAGA temporal transformer before the adapter/PCH head. Stage 8.5 showed that this conflicts with the current paper narrative, which requires YawDD to use the same pooled PCH architecture family as AIDE.

The files are preserved as historical reference only. Checkpoints, JSON files, logs, and diagnostic reports are intentionally kept and were not deleted.
