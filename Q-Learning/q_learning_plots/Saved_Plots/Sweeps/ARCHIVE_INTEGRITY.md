# Saved sweep archive integrity

On October 30, 2026, all 19 remaining gzip run archives were checked by streaming
their full decompressed contents through Python's gzip reader. Eighteen passed,
including the nine byte-identical relocated overnight archives and the updated
top-level `Q-Learning/q_learning_run.pkl.gz`.

The archive at
`efficient_control_sweeps/20261005-224846/gamma_0.9975/q_learning_run.pkl.gz`
failed with `Error -3 while decompressing data: too many length or distance symbols`.
It remains local and is excluded by a specific `.gitignore` rule. Its training
and plotting logs remain included for diagnosis. Passing a gzip check validates
the container's integrity; it does not establish correctness of training results.

See [archive_integrity.json](archive_integrity.json) for per-file checks. The 19
files relocated from `Q-Learning/overnight_runs/20261002-011558/` were also verified
against their original Git blob hashes. The 1,314 removed overnight plots remain
recoverable from repository history before commit `2a08246`.
