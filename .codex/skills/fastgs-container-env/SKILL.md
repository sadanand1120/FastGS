---
name: fastgs-container-env
description: Use the FastGS runtime context for execution commands that depend on the container/env (container `fresh`, conda env `fastgs`, repo `/robodata/smodak/repos/FastGS`). Host-side repo inspection is fine for simple lookups and search.
---

# FastGS Container + Env Execution

Use this workflow for execution commands that need the project runtime. Do not use it for simple host-side inspection of repo-mounted files such as `rg`, `grep`, `sed`, `ls`, or `git diff`.

1. Enter container: `docker exec fresh ...`
2. Ensure env context is `fastgs` (path: `/opt/miniconda3/envs/fastgs`)
3. Change to repo: `cd /robodata/smodak/repos/FastGS`
4. Run the requested command

Canonical command wrapper:

```bash
docker exec fresh bash -lc '
  source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate fastgs &&
  cd /robodata/smodak/repos/FastGS &&
  <COMMAND>
'
```

If `conda activate` is unavailable, fall back to:

```bash
docker exec fresh bash -lc '
  cd /robodata/smodak/repos/FastGS &&
  /opt/miniconda3/envs/fastgs/bin/python -V
'
```

## Troubleshooting

- If container access fails: verify `fresh` is running.
- If env activation fails: verify `/opt/miniconda3/envs/fastgs` exists in the container.
- If runtime imports mismatch: print `which python`, `python -V`, and any relevant package versions from inside the container before proceeding.
