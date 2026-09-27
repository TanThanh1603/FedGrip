# Native FedSAM and StableFDG

These are client/server integrations into this project's FDG protocol, not claims
of bitwise reproduction of published scores. Existing FedGRIP/FedOMG and `out/`
are unchanged. Neither integration requires imports from the vendored frameworks.

## Running

Both methods are registered in `main.py`. For a common supported backbone:

```bash
python main.py FedSAM -d pacs --model res18 --shared-partition-seed 40 --round 100 --num_epochs 3 --sam_rho 0.1
python main.py StableFDG -d pacs --model res18 --shared-partition-seed 40 --round 100 --num_epochs 3
```

These commands run domains sequentially, using the existing partition and HTML
logging pipeline. Add `--only-test-domain photo` to run one PACS domain. They do
not add OMG or GRIP plugins. Use identical backbone/pretrained weights, optimizer,
partition, seed and training budgets when making comparisons. StableFDG supports
`res18` (default) and `res50`; native generic models also support `res18` now.
StableFDG intentionally rejects MobileNet rather than
silently substituting a different method. StableFDG uses ImageNet V1 weights as
in its upstream ResNet implementation. The example `res18` commands both use V1.
The existing generic `res50` uses Torchvision DEFAULT (V2), so comparisons with
`res50` require explicitly aligning pretrained weight choice first.

## FedSAM provenance

Reference: https://github.com/skydvn/FedOMG at
`e1ddccabd1f4aab1443c117b472f4a5d65a74ad2`,
`FedOMG-DG/algorithms/fedsam/optimizer/esam.py` and `utils/trainval_func.py`.
Two differentiated passes, perturbation `rho * grad / (norm + 1e-7)`, restoration,
gradient clipping at 10, then the base optimizer step. The extra upstream
metrics-only training-mode forward is omitted (therefore BN/RNG trajectories
are not bitwise identical). Project optimizer and cosine schedule are retained;
`sam_rho=0.1` is the source script default, not a claim of paper-tuned settings.

## StableFDG provenance and integration choices

Reference: https://github.com/savertm/StableFDG_github at
`77eac350f8515c45630f60d2564863e3da8c8687`:
`ops/style_insert.py`, `ops/oma.py`, `ops/cross_attn.py`, ResNet backbone and
`TrainerX_fed`/`Vanilla2`.

- Style sharing: collect mean/std distribution of layer1 styles using local
  training data only; assign a different peer client; enable after round 0.
  KMeans local centers plus Gaussian peer styles; probability 0.5. At short
  batches, use min(16, batch_size // 2) shared styles.
- Style exploration at layer1/2/3: probability 0.5, Beta(0.1, 0.1) mixing,
  class-balanced oversampling, default exploration 3 and 32 extra samples.
- AFH: same-class query pairing, singleton support from local class-balanced
  examples, spatial attention, concatenated pooled/highlighted features and
  doubled classifier input. Query/key norms have a numerical epsilon; average
  branch uses actual spatial size (equals upstream division by 49 for 7x7).
- Explicit label lookup works with missing/non-contiguous local classes.
  Supplemental samples are drawn once per local batch, one per local class;
  smallest-class oversampling ties are randomized one sample at a time.
- Style collection uses eval mode and streaming statistics without changing BN
  buffers. Native client optimizers persist like other project baselines rather
  than being recreated for every global round as in upstream StableFDG.
- Project source partitions, full client participation, sample-mass aggregation,
  preprocessing and optimizer/scheduler are retained. The upstream 30-client,
  1/3-participation setup, uniform aggregation and original augmentation protocol
  are not reproduced by this integration.

## Verification

```bash
python -m unittest tests.test_native_fdg_baselines -v
```

CPU tests cover SAM/reference update agreement, exception restoration, style
sharing and short batches, oversampling/label alignment, AFH singleton labels,
ResNet train/backward/eval, native client collection/training, and peer exclusion.
No full dataset training or published accuracy reproduction is implied.
