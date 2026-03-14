# Dense CLIP Optimization Program

This is the specialized autonomous-optimization program for `custom/train_dense_clip_langsplat_autoencoder.py`.

## Setup

Before any experiment loop:

1. Read these files fully:
   - `custom/README_dense_clip_optimization.md`
   - `custom/benchmark_dense_clip_autoencoder.py`
   - `custom/train_dense_clip_langsplat_autoencoder.py`
2. Verify `custom/trial_images/` exists.
3. Follow repo runtime policy from `AGENTS.md`:
   - execute GPU/runtime commands in container `fresh`
   - activate conda env `fastgs`
   - `cd /robodata/smodak/repos/FastGS`
   - set `CUDA_VISIBLE_DEVICES='1'`
4. Initialize `custom/dense_clip_optimization_results.tsv` with this header if it does not exist:

```tsv
commit	profile	best_eval_mse	extract_s	cache_s	train_s	export_s	total_s	projected_1000_s	peak_vram_gb	status	description
```

5. Establish the baseline:
   - run `smoke`
   - run `quality`
   - log the `quality` result as `baseline`

## Scope

Default mutable target:
- `custom/train_dense_clip_langsplat_autoencoder.py`

Read-only files:
- `custom/benchmark_dense_clip_autoencoder.py`
- `custom/program_dense_clip_optimization.md`
- `custom/README_dense_clip_optimization.md`

If advanced optimization truly requires a tiny helper file in `custom/`, that is allowed, but only after simpler options are exhausted and the helper remains generic. Do not touch the benchmark harness.

## Objective

Primary objective:
- reduce end-to-end runtime, especially projected runtime for 1000 images

Constraint:
- do not materially worsen `best_eval_mse`

The benchmark dataset is small, so do not game it. Every change must still make sense for arbitrary scenes with roughly 1000 images.

## Hard Constraints

- No special-casing `custom/trial_images`, specific image stems, exact image counts, or exact grid sizes.
- No fast paths that silently change semantics for larger scenes.
- No deleting required outputs just to benchmark faster.
- No benchmark-specific branches in the target script.
- No hacks that rely on the benchmark always using 15 images.

## Optimization Ladder (suggestive, not prescriptive)

Work in this order unless profiling clearly proves otherwise:

1. Easy:
   - remove redundant copies and conversions
   - reduce Python/Numpy overhead
   - better batching/grouping
   - reuse allocations
   - avoid unnecessary synchronization
   - reduce repeated file opens / metadata scans
   - improve mmap/cache usage

2. Medium:
   - overlap host/device work where safe
   - tune block sizes and batch sizes
   - use `torch.compile` only where stable and measurable
   - mixed precision for safe subpaths
   - improve checkpoint/log writing overhead if it is measurable

3. Advanced:
   - fused kernels / Triton / custom CUDA only if profiling shows a real generic bottleneck
   - any such kernel must remain shape-generic and scene-generic

## Experiment Loop

For each experiment:

1. Inspect current git state.
2. Make one coherent change.
3. Commit the change before benchmarking.
4. Run `smoke`:

```bash
python custom/benchmark_dense_clip_autoencoder.py --profile smoke > run_smoke.log 2>&1
```

5. If `smoke` crashes, inspect `run_smoke.log`, fix or discard.
6. If `smoke` succeeds, run `quality`:

```bash
python custom/benchmark_dense_clip_autoencoder.py --profile quality > run_quality.log 2>&1
```

7. Extract the summary from `run_quality.log` and append a TSV row.
8. Keep the commit only if it is a real win.
9. If not a win, revert to the previous kept commit.

## Keep / Discard Guidance

Compare against the current kept baseline, not just the immediately previous run.

Keep when:
- `best_eval_mse` is not worse by more than about 2%, and
- projected 1000-image runtime improves clearly, or
- quality improves materially at similar speed, or
- code gets simpler while preserving both speed and quality

Discard when:
- quality regresses clearly
- speedup is tiny and complexity cost is high
- the change only helps the tiny benchmark in a suspicious way

## Logging

Log only the `quality` run in the TSV unless `smoke` crashes and the crash itself is worth recording.

Suggested status values:
- `keep`
- `discard`
- `crash`

## Notes

- Always prefer non-interactive git commands.
- Always check for stray processes if a run may still be alive.
- Never pause the loop to ask whether to continue once the human has explicitly started autonomous benchmarking.
