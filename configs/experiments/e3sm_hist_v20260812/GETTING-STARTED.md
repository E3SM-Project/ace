# Getting started — aug26 historical campaign

For Finn, Olawale and Naser. Two lanes: **Lane A** launches one run by hand and
is the one to read first; **Lane B** drives the whole 35-run campaign. Both end
up running the same code against the same configs.

Deeper background: [README.md](README.md) for the configs and their gotchas,
and [EXPERIMENTS.md](EXPERIMENTS.md) for the campaign design and the
measurements.

---

## TL;DR

```bash
# once
git clone <repo> && cd ace && git checkout e3sm/exps/hist-v2026.8.0
uv sync --frozen                                   # ~1 min
wandb login                                        # YOUR key, not anyone else's
cd configs/experiments/e3sm_hist_v20260812
./stage-shared-data.sh --check                     # inputs where the configs expect

# every time
./sbatch-scripts/run-train.sh atm E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S01
```

That prints a job id, emails you at `$USER@nersc.gov` on start/end/fail/requeue,
and writes `joblogs/<runid>-<jobid>.out`. Everything else on this page is detail.

---

## Before you launch anything

| check | command | expected |
|---|---|---|
| on the right branch | `git rev-parse --abbrev-ref HEAD` | `e3sm/exps/hist-v2026.8.0` |
| tree is clean | `git status --short` | no output |
| venv built | `ls .venv/bin/torchrun` | exists |
| inputs readable | `./stage-shared-data.sh --check` | `all inputs readable` |
| wandb is you | `wandb login` then check the printed user | your account, entity `e3sm-aig` |
| configs agree with their names | `./check_campaign.py` | `35 configs, 0 with problems` |

**Your W&B key is yours.** Do not paste one into a config, the repo, the wiki or
Slack. Everyone logs in with their own key and runs still land in the shared
`e3sm-aig/SamudrACE-E3SMv3` project, because entity and project come from the
config, not from the credential.

---

## Lane A — one run, by hand

### A1. Pick a run

```bash
column -t runs/MANIFEST.tsv | head -8      # priority, runid, realm, nodes, ranks
```

A run id is `<exp>.<date>.<realm>.<tuning_set>.S<seed>` and the tuning set
`A?_B??_C?_L?_O?_W?_X?` is the complete statement of what varies. `E01…S01` is
the atmosphere baseline; `E11…S01` is the ocean baseline.

### A2. Dry-run it

```bash
./sbatch-scripts/run-train.sh atm E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S01 --no-submit
```

Stages the config to `$PSCRATCH/fme-config/<uuid>`, sources the run's `.env`,
applies its node count and runs the typed validator — everything except the
`sbatch`. If this passes, the queued job will start.

### A3. Submit

```bash
./sbatch-scripts/run-train.sh atm E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S01
```

During the hackathon window prefix `RESERVATION=_CAP_aigs_hist`, or the job sits
in the general queue while 96 reserved nodes idle:

```bash
RESERVATION=_CAP_aigs_hist ./sbatch-scripts/run-train.sh atm <runid>
```

### A4. Watch it

```bash
squeue -u $USER -o "%.10i %.45j %.2t %.10M %.6D %R"
grep -m1 -A12 "=== run" joblogs/<runid>-<jobid>.out    # identity banner
grep "Time taken for epoch" joblogs/<runid>-<jobid>.out
```

The job log opens with a banner naming the run id, the job, the node list, the
config, the output directory, the commit and the wandb run. Grep `=== run`.

Then W&B: `e3sm-aig / SamudrACE-E3SMv3`, filter by tag (`aug26`, `E01`, `atm`,
`C1`, …). The per-epoch cost is logged as `epoch_train_seconds`,
`epoch_validation_seconds`, `epoch_inference_seconds`, `epoch_total_seconds` —
use those, not a stopwatch.

### A5. Know what "finished" looks like

* `REAL_EXIT=0` in the job log, and `DONE ---- rank 0`.
* **Ignore the traceback wall at the end of a successful run.** `time_buffer`
  teardown prints `ValueError: semaphore or lock released too many times` and
  `OSError: [Errno 9] Bad file descriptor` from `QueueFeederThread` on clean
  exits. Judge the run by `REAL_EXIT`, never by the log tail.
* Checkpoints land in `<output>/training_checkpoints/`: `ckpt_NNNN.tar` (7.3 GB,
  atmosphere) and `ema_ckpt_NNNN.tar` (1.8 GB) per epoch, plus `ckpt.tar`,
  `best_ckpt.tar`, `best_inference_ckpt.tar`.

### A6. If it dies

| symptom | cause | fix |
|---|---|---|
| job never starts, no log at all | submitted from a cwd with no `joblogs/` | already fixed by `--chdir`; if you called `sbatch` directly, don't |
| every rank `invalid device ordinal` | `FME_USE_SRUN=1` | never set it; torchrun sets the device from `LOCAL_RANK` |
| `No batches in dataloader: 0 samples` | a window shorter than `~11 * batch_size` timesteps | widen it |
| `UnionMatchError: can not match type "list"` | IC count not divisible by rank count | run `--no-submit` first, which catches it on a login node |
| ranks time out on the TCPStore | rendezvous host is not a node of *this step* | see "two runs in one allocation" below |

Resume is automatic — a requeued segment picks up from
`<output>/training_checkpoints/ckpt.tar`, which is written at every epoch
boundary, and from a mid-epoch restart checkpoint when it has one. **It skips
the batches already done inside the current epoch** — job 57761772 resumed with
`skip first 148 batches since these were already processed for this epoch`, then
ran the remaining 8,069 of the epoch's 8,217. Getting there took three
tries, and the first two walltime requeues on 2026-08-30 produced no mid-epoch
checkpoint at all. Job 57758390: `--signal=USR1@120` without a `B:` prefix reaches the
python ranks, which have no SIGUSR1 handler, so they died at default
disposition. Job 57759729: with `B:` the routing was right, but the batch trap
stopped the step with `kill -TERM` on srun, and SIGTERM is one of the few
signals srun does not forward — it aborted the step and all 16 ranks came back
`Killed`, so no handler ran anywhere. The trap now uses
`scancel --signal=TERM`, which delivers a real SIGTERM inside the step's
cgroup, and the lead time is `--signal=B:USR1@300`. Job 57760702 confirmed that
much — `REAL_EXIT=143`, `Exited with exit code 143`, no `Killed` — and then
failed to checkpoint for a third reason: the same cgroup-wide SIGTERM kills the
DataLoader workers, and torch answers their deaths by raising
`RuntimeError: DataLoader worker ... is killed by signal` in the main thread,
which landed inside `destroy_process_group` and then inside the checkpoint's
`get_state`. Both are wrapped against exceptions, so the rank exited 143
looking clean with its collectives still up. Fixed in
`fme/core/distributed/shutdown.py` by resetting SIGCHLD before the teardown.

**Budget a requeue at the dataset setup and the queue wait — about 21 min on
CFS — not at the partial epoch.** The margin is comfortable: torchrun's agent
SIGKILLs the ranks 30 s after the signal reaches it, and on 57761772 the
teardown took 587 ms and the 6.8 GiB restart checkpoint 10.4 s. (The 31.1 s in
`EXPERIMENTS.md` is the whole per-epoch write of ~20 GB, EMA and epoch-numbered
copies included, not this one file.) To resume an ad-hoc run by hand:
`RESUME_JOB_ID=<jobid> ./sbatch-scripts/run-train.sh atm`.

### Two guards that will stop you, on purpose

**"refusing to submit from a dirty worktree."** `.venv` is an editable install
pointing straight at the checkout, so a job runs whatever is in the tree *when
it starts*, not the SHA in `COMMIT`. Editing between submit and start — or
before a requeued segment — silently changes the code mid-run. Commit or stash.
`FME_ALLOW_DIRTY=1` overrides it and prints a warning instead; use that for
scratch runs, never for a campaign run.

**"`<output>` already holds a checkpoint."** The trainer restores
`ckpt.tar` if it finds one, whatever the newly staged config says. So a second
submission of a run id continues the first run rather than starting over — and
if two people submit the same id, one silently continues the other's run under
a different config. To continue deliberately:

```bash
./sbatch-scripts/run-train.sh atm <runid> --resume
```

If the staged config's SHA-256 or the recorded commit differs from what that
output directory was produced with, `--resume` is refused too, and you need
`--force` to say "yes, make this one run id two experiments". The hash is in the
job banner (`config sha`) and in `<output>/job_config/CONFIG_SHA256`.

---

## Lane A′ — interactive, for debugging only

Use this to see a stack trace live, not to do science.

```bash
salloc --nodes 4 --qos interactive --time 04:00:00 \
       --constraint gpu\&hbm80g --account=e3sm_g --gpus-per-node=4

# inside the allocation, from the repo root
srun --nodes 4 --ntasks-per-node 1 --gpus-per-node 4 \
  .venv/bin/torchrun --nnodes 4 --nproc_per_node 4 \
    --rdzv-backend=c10d --rdzv-endpoint="$(hostname):29507" \
    -m fme.ace.train configs/experiments/e3sm_hist_v20260812/runs/<runid>.yaml \
    --override experiment_dir=$PSCRATCH/fme-output/scratch-$USER
```

Single node is simpler — `torchrun --standalone --nproc_per_node 4`.

Three things that bite here:

1. **Source the run's `.env` first**, or the run lands in W&B unnamed and
   someone has to delete it:
   `set -a; . configs/experiments/e3sm_hist_v20260812/runs/<runid>.env; set +a`
2. **`srun` dies with the launching session.** Anything multi-hour goes through
   `sbatch`, not an `salloc` you might disconnect from.
3. **Two runs in one allocation**: the c10d rendezvous host must be a node of
   *that step*, and with `srun --nodes 2` inside a 4-node `salloc` the
   allocation's first node need not be in it. Compute it inside the step —
   `MASTER_ADDR=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1)` —
   and give each step its own port. Otherwise every rank times out on a
   TCPStore it cannot reach, 60 s at a time, with no other explanation.

A 20-minute end-to-end smoke test that touches every code path:

```bash
uv run python make_smoke_config.py config-train-ocn.yaml $PSCRATCH/smoke-ocn.yaml \
    --experiment-dir $PSCRATCH/smoke-out
uv run torchrun --nproc_per_node 4 -m fme.ace.train $PSCRATCH/smoke-ocn.yaml
```

---

## Lane B — the whole campaign

```bash
./sbatch-scripts/generate-campaign.sh --list     # 35 runs, 129 nodes
./sbatch-scripts/generate-campaign.sh            # writes runs/ + MANIFEST.tsv, then checks
./sbatch-scripts/submit-campaign.sh --preflight  # stage + validate all 35, queue none
./sbatch-scripts/submit-campaign.sh --dry-run    # print what would be queued
RESERVATION=_CAP_aigs_hist ./sbatch-scripts/submit-campaign.sh --max-priority 1
```

Submit in priority order and stop early:

| flag | effect |
|---|---|
| `--max-priority 1` | the four bolded baselines only — 14 nodes |
| `--max-priority 3` | + science ablations + extra seeds — 84 nodes, fits the reservation |
| (none) | all 35 — 129 nodes, so 45 nodes' worth queues behind |
| `--only atm` / `--only E05` | one realm or one experiment |

### Do not skip the ramp

**Monday morning is `--max-priority 1`, then wait.** The window has a 1.4x
margin and contention was measured at up to 2.1x with one competing job, so
releasing 129 nodes at once is the one mistake that costs the campaign rather
than a run.

1. Queue P1 (14 nodes, the four baselines everything is compared against).
2. Let E01 log **two** epochs — epoch 1 carries setup and the first checkpoint.
3. Read `epoch_total_seconds` in wandb:
   * **< 10,800 s** — release P2+P3 with `--max-priority 3`, P4 as nodes free.
   * **10,800–11,900 s** — release P2+P3, skip P4.
   * **> 11,900 s** — stop. 30 epochs no longer fits; cut a lever from
     EXPERIMENTS.md "The levers, if the budget gets tight" first.

### Where your output goes, and the one rule tooling cannot enforce

`CAMPAIGN_ROOT` defaults to **`$PSCRATCH/aug26` — your own scratch**, and that is
the decision, not a placeholder. Three quotas beat one, and nobody can purge or
overwrite anyone else's checkpoints.

The catch: every guard is scoped to you. `run-train.sh` refuses to submit a run
id that already has a checkpoint in *your* `$PSCRATCH`, refuses one whose config
or commit disagrees with what already ran there, and refuses one already sitting
in *your* `squeue`. **None of them can see another person's scratch or queue.**
Two people submitting the same run id get two independent trainings reporting to
the same wandb name, and nothing warns either of them.

So: **every run id has exactly one owner, and only its owner submits it.** Agree
the split before Monday and treat it as binding. wandb is the shared surface —
all 35 runs report to one project wherever they physically run.

`runs/` is entirely generated. Never hand-edit a file in it: regenerate, or the
next `generate-campaign.sh` silently reverts you.

**It is also identical whoever generates it** — no username, no scratch path, no
timestamp. So regenerating against the committed `runs/` is a no-op: your
worktree stays clean, `run-train.sh` is happy, and **you never have to commit
anything to launch a run.** Identity is attached at submit time instead —
`run-train.sh` appends `owner $USER | out $CAMPAIGN_ROOT/$RUNID` to the wandb
notes, so wandb records who actually submitted rather than who generated.
`check_campaign.py` fails any generated file that names a `/pscratch/` path,
which is what keeps this true.

The one thing that *does* need committing is a change to the run list itself —
`make_ablation_config.py`, the baselines, or the checker. That is a change to
the campaign, so it goes through a normal commit and everyone regenerates.

### Queue facts worth knowing before you submit

* `MaxJobsAccruePU = 2` on `gpu_regular`: only two of your jobs accrue age at a
  time. The rest sit frozen at the QOS floor. Submitting all 35 at once does not
  buy queue position — priority order does.
* Age is worth exactly 1 point/minute, and `bf_min_prio_reserve = 69121` is
  24 hours of accrual above the floor. Below it a job gets no reserved start
  time and runs on opportunistic backfill only.
* Fairshare and job-size weights are both **0**. There is no penalty for
  submitting early and no bonus for being small.
* The configs require `-C gpu&hbm80g`: 256 of Perlmutter's 1,664 GPU nodes.
  The reservation is 96 `hbm80g` nodes, i.e. 37% of that pool — which is why
  running inside it matters so much more than queue tactics.

---

## Email

On by default: `$USER@nersc.gov` on `BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90`.

```bash
FME_MAIL_USER=you@lbl.gov  ./sbatch-scripts/run-train.sh atm <runid>
FME_MAIL_TYPE=FAIL,TIME_LIMIT_90 ./sbatch-scripts/submit-campaign.sh   # trouble only
FME_MAIL_TYPE=NONE ./sbatch-scripts/run-train.sh atm <runid>
```

`REQUEUE` is the one that matters — it is the difference between "still
training" and "has been requeueing all night". The full campaign is order 250
messages, so filter on the subject, which carries the job name and id.

Retrofit an already-queued job without resubmitting:

```bash
scontrol update JobId=<id> MailUser=$USER@nersc.gov \
    MailType=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
```

---

## W&B: what gets uploaded, and what stays on disk

The team account has **100 GB**. Measured 2026-08-30, the committed setup was
uploading one PNG per channel per map metric per block per epoch — 506 images
per atmosphere epoch, 484 per ocean epoch, order 50 GB across the campaign in
maps alone. That is now narrowed.

| | where it lives |
|---|---|
| every scalar — `rmse/<var>`, `bias/<var>`, `rmse/channel_mean`, `mean/*`, losses, `epoch_*_seconds` | W&B, for **every** channel |
| 1D plots — annual means, power spectra, histograms, ENSO index | W&B, as interactive plotly charts (21 KB each) |
| time-mean and bias maps | W&B, for the plotted channels |
| **everything, every channel, full fields** | `<output>/` as netCDF, every epoch |

**What "the plotted channels" means:** 38 of 50 in the atmosphere, 28 of 80 in
the ocean. Every flux and every surface/2m/10m field is in; what is dropped is
the interior levels — `T`/`STW`/`U`/`V` indices 2, 3, 5 in the atmosphere, and
the coarsened stacks' levels 3–8 and 10–16 in the ocean. One list drives the
maps, the spectra and the histograms, so a single screen shows the same channels
three ways.

Nothing is lost: `save_per_epoch_diagnostics: true` writes every aggregator's
full fields to the output directory as netCDF each epoch. Want a map of
`T_3`, or `salinityCoarsened_11`? Open
`<output>/output/inference/epoch_NNNN/time_mean_diagnostics.nc` and plot it, at
whatever colour scale and projection you like.

Measured on one epoch with both blocks firing: **228 map PNGs (27 MB)** and
**272 plotly charts (5.7 MB)**. At production cadence that is ~20 MB/epoch for
the atmosphere and ~8 MB for the ocean — **about 26 GB of the 100 GB account**
for the whole campaign.

One gotcha when you look at a short test run: `annual` and `enso_index` need
more than two years of rollout and are silently skipped below that, so a
quick smoke test looks far emptier than a production run will.

Three rules that follow:

* **Do not widen your own run's plot list.** `check_campaign.py` fails a config
  whose four lists disagree, and the cost of widening is shared across a 100 GB
  account.
* If you add an aggregator, check whether it emits one figure per variable
  before you submit 25 runs with it on.
* One image metric has no per-channel control — the validation ensemble's mean
  maps, 174 PNGs per atmosphere epoch. It is off. Ask before turning
  `ensemble_denorm.log_mean_maps` back on; it is all-or-nothing.

---

## What a run costs

Measured 2026-08-30 on 4x A100-80GB; see EXPERIMENTS.md "Measurements" for the
full breakdown and the wandb runs.

| | atmosphere (E01) | ocean (E11) |
|---|---|---|
| nodes | 4 | 2 |
| dataset setup, per job start **and per requeue** | **22.4 min** | **14.0 min** |
| training | 0.899 s/batch, 8,217 batches/epoch | 1.34 s/batch, 411 batches/epoch |
| inline inference, per scored rollout | 35–42 min | 4.9 min |
| epoch total, everything included | **2.8–3.0 h** | **0.32 h** |
| epochs | 30 | 150 |
| whole run, setup included | **88–92 h** against a 126 h window | **~49 h** |
| checkpoints accumulated per epoch | 9.1 GB | 1.7 GB |
| whole run on disk | ~275 GB | ~250 GB |

Four consequences for how you work:

* **Setup is 14–22 minutes and is paid again on every requeue.** Do not chase a
  run's first twenty minutes of silence; it is opening 1,501 netCDF files, and
  the cost does not depend on how much of the record the run actually reads.
  Yes, a `$PSCRATCH` copy of the inputs exists and opens ~9x faster — and it is
  8–14% slower per step, which nets 3–6 h worse over a 30-epoch run.
  **Do not repoint your run at it**; see EXPERIMENTS.md "The filesystem A/B".
* **Inference is a quarter to a third of an epoch**, and it is what the older
  63 h / 24 h estimates left out. Judge progress by `epoch_total_seconds`.
  The same correction is why the ocean's O1 arm (E17) keeps 30 epochs: priced
  with inference, 30 O1 epochs and 150 O5 epochs are both ~49 h.
* **Watch E01's first three epochs.** Above ~3.3 h/epoch a 30-epoch run does not
  finish inside the reservation; EXPERIMENTS.md "The levers" says what to cut.
* **Disk grows fast** — ~9.2 TB for the campaign. Check `myquota` from a
  **login** node; it fails on a compute node.
