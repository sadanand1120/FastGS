# Dense CLIP Autoencoder Optimization

This setup is the `custom/` adaptation of the ideas from `custom/autoresearch`, specialized for optimizing [train_dense_clip_langsplat_autoencoder.py](/robodata/smodak/repos/FastGS/custom/train_dense_clip_langsplat_autoencoder.py).

The goal is narrow:
- reduce end-to-end runtime
- keep reconstruction quality roughly intact
- avoid hacks that only work for `custom/trial_images`
- preserve scalability to scenes on the order of 1000 images

## Files That Matter

- `train_dense_clip_langsplat_autoencoder.py`
  This is the main target. It is the file the optimization agent should modify by default.

- `benchmark_dense_clip_autoencoder.py`
  Fixed benchmark harness. It runs the target script with a fixed profile, captures a structured summary, projects runtime to 1000 images, and keeps raw logs in `custom/dense_clip_optimization_logs/`.

- `program_dense_clip_optimization.md`
  The agent operating instructions. This is the use-case-specific replacement for `custom/autoresearch/program.md`.

- `analyze_dense_clip_optimization_results.py`
  Lightweight results summarizer for the benchmark TSV log.

## Benchmark Philosophy

The benchmark uses `custom/trial_images` because it is fast enough for many iterations, but the instructions explicitly forbid overfitting to it.

The benchmark therefore emphasizes:
- stage timings: extraction, cache build, training, export
- normalized throughput via `num_feature_vectors`
- projected runtime for 1000 images
- held quality metric from the target script: `best_eval_mse`

This is not perfect, but it is much better than timing only the tiny 15-image trial set and calling it done.

## Fixed Profiles

- `smoke`
  One epoch. Used as a quick crash/perf sanity check.

- `quality`
  Ten epochs. Used for keep/discard decisions.

Both profiles force a full end-to-end run:
- re-extract features
- rebuild the fp16 cache
- train
- export low-D maps

The benchmark disables PCA visualization during measurement because that is not part of the optimization target.

## Rules For Future Optimization

- Do not special-case `custom/trial_images`, fixed image counts, fixed stems, or fixed patch grids.
- Do not disable outputs or correctness checks that the real pipeline needs.
- Prefer changes that reduce Python overhead, copies, synchronization, or generic I/O/computation bottlenecks.
- Start with easy improvements first, then medium, then advanced kernels only if profiling justifies them.
- If an optimization requires extra files, keep them small and generic. The benchmark harness and this README are read-only.

## Runtime Context

Per repo policy, runtime commands should be executed in the `fresh` container, in the `fastgs` conda env, from `/robodata/smodak/repos/FastGS`, with `CUDA_VISIBLE_DEVICES='1'` set for GPU runs.

Canonical benchmark example:

```bash
docker exec fresh bash -lc '
  source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate fastgs &&
  export CUDA_VISIBLE_DEVICES="1" &&
  cd /robodata/smodak/repos/FastGS &&
  python custom/benchmark_dense_clip_autoencoder.py --profile smoke
'
```
