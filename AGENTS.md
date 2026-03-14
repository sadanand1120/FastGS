# FastGS Agent Notes

## Runtime Context
- Run runtime-dependent project execution inside the container, in the conda environment. Do not run project scripts, training, rendering, metrics, debugging commands, or environment-sensitive checks on the host.
- Container name: `fresh`
- Conda environment: `fastgs`
- Conda env path: `/opt/miniconda3/envs/fastgs`
- Repo path inside workflow: `/robodata/smodak/repos/FastGS`

## Command Execution Policy
- For project execution commands, execute in container + env context:
  1. `docker exec` into `fresh`
  2. activate `fastgs`
  3. `cd /robodata/smodak/repos/FastGS`
  4. run the command
- For any command that uses GPU, set `CUDA_VISIBLE_DEVICES='1'` before running the command.
- Do not create lingering or stray processes. After running commands that may stay alive, explicitly check whether any process you started is still running; if it is, kill it cleanly before finishing.
- For repo-mounted source files, prefer host-side edits and host-side read-only inspection by default. Use container-side editing only when a runtime-dependent workflow explicitly requires it.
- Non-execution inspection on repo-mounted files can be done on the host. This includes simple file lookups, `rg`, `grep`, `sed`, `ls`, `git diff`, and similar read-only inspection under `/robodata/smodak/`.

## Project Shape
- FastGS is a Gaussian Splatting codebase with training, rendering, and evaluation entry points rooted in `train.py`, `render.py`, and `metrics.py`.
- `custom/` contains local experimental scripts and one-off utilities that may not match the main training pipeline exactly.
