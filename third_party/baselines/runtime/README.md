# Runnable baseline adapters

This directory contains runnable copies of the pinned sources in `../sources`.
The local changes only provide configurable data
and result paths, PyTorch 2.5-compatible optional imports, the omitted PerAvg
entry point, the omitted FedPAC `lamda` argument, and correct CNN dimensions
for the four image datasets. Its inclusive `rounds + 1` loops are corrected
so `--rounds 50` performs exactly 50 communication rounds. The untouched upstream files and their hashes
remain in `../sources` for provenance checks.
