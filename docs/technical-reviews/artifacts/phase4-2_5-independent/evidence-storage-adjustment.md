2026-10-05: Before any 2.5 report was written, this reviewer had created three
6/8/10 KiB local checkpoint snapshots. The coordinator requested references to
the existing checkpoint paths and SHA-256 instead. Only the three snapshots
created by this reviewer are removed. Existing checkpoints are unchanged.

The probe now loads the original existing local CPU checkpoints, checks their
SHA-256 against that run's raw result.json, and permanently records the paths,
hashes, all evaluated losses, target denominators, code, commands and environment.
The published 26f34eb run and existing local 5d60e35 CPU run remain separate;
no fresh training or original 26f34eb checkpoint reproduction is claimed.
