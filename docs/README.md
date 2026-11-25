# eforecast documentation

Six documents. Find yourself in the first column.

| if you are… | read |
| --- | --- |
| setting up a machine, or just want to use the lab from your own computer | [installation.md](installation.md) |
| running a case: configuring it, starting it, reading the result | [user.md](user.md) |
| running the installation: services, images, approvals, backups, venues | [administrator.md](administrator.md) |
| working from a terminal | [cli.md](cli.md) |
| stuck, or something failed | [troubleshooting.md](troubleshooting.md) |
| trying to understand what the pipeline actually does | [how_it_works.md](how_it_works.md) |

If you are new and have twenty minutes, read [how_it_works.md](how_it_works.md) first. Everything
else makes more sense afterwards.

## Before copying a terminal command

Load the installation's own settings, so commands answer about the lab rather than about your
shell:

```bash
set -a; . /etc/eforecast/webui.env; set +a
```

`bin/eforecast` does this for itself for seven of the settings, when you have not set them, and
says which it took; [cli.md §4](cli.md#4-the-settings-that-matter) says which seven, and what
every setting is for. The spelled-out `python -m eforecast.config.…` form takes none.

`<case>` means a **path** after `--config` in a `run_*.py` script (a YAML file, a parts directory
or its `main.yaml`), and a **workspace case name** after a `bin/eforecast` verb.

## Deeper reference

The six documents above are self-contained. These go further on one subject each, and are kept
because the detail in them does not compress:

| file | subject |
| --- | --- |
| [how_it_works.md](how_it_works.md) | the whole workflow — start here |
| [compute_workers.md](compute_workers.md) | work that runs in a process of its own: what it is given, where it writes, how it ends |
| [cpu_training_scopes.md](cpu_training_scopes.md) | the shallow CPU families, their scopes, trial generations, and studies side by side |
| [gpu_training_optuna.md](gpu_training_optuna.md) | GPU training, Optuna, Docker and CPU studies, with a runbook |
| [deep_models.md](deep_models.md) | the networks themselves: every layer and when to use it, how branches for different data are joined, the data tags, the training loop, the cross-site graph |
| [combine_evaluation.md](combine_evaluation.md) | prediction, selection, combination, calibration and evaluation in detail |

Elsewhere in the repository: `deploy/` for the deployment units, `webui/README.md` for frontend
development, `docker/` for the images and the monitoring stack. Running a case on a Kubernetes
cluster — and trying that on this machine first, with one script — is in
[deploy/kubernetes/README.md](../deploy/kubernetes/README.md).

## What used to be here

This folder previously held about 260 further documents: a per-task record for every piece of
review work, a per-step record for every implementation step, the review plans, and a 661 MB tree
of machine-readable evidence. They described **how the work was done**, which is a different
question from how the pipeline works, and they had grown far past the point where anyone could
find anything in them.

They are all in the git history. `git log --diff-filter=D --name-only -- docs/` lists them, and
any one of them can be brought back with `git show <commit>:<path>`.

Two things that lived in that tree were *not* documentation and have moved into the test suite,
where they belong: `tests/probes/` (a helper that builds the small case six test files use) and
`tests/data/` (the recorded estimator inventory and the scikit-learn equivalence evidence a test
asserts against).
