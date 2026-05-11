# ACE E3SM Training on Perlmutter

## Batch job submission

```bash
./run-train-perlmutter.sh
```

This copies the config to a staging area under `$PSCRATCH/fme-config/` and submits
`sbatch-scripts/sbatch-train.sh` to the `gpu_regular` queue under account `e3sm_g`.

To resume a failed job, set `RESUME_JOB_ID` in `run-train-perlmutter.sh` before submitting.

---

## Interactive runs (for testing)

### Step 1 — Get an allocation

The model requires 80 GB GPUs (`hbm80g`), so use `gpu_regular` (supports `hbm80g`).
Note that `gpu_interactive` does not support the `hbm80g` constraint.

```bash
salloc -A e3sm_g -q gpu_regular -C "gpu&hbm80g" \
  --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
  -t 04:00:00
```

This queues like a normal batch job — once nodes are allocated you get an interactive shell.

For a quick config/container sanity check on a single node, `gpu_debug` queues faster:

```bash
salloc -A e3sm_g -q gpu_debug -C "gpu&hbm80g" \
  --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=128 \
  -t 00:30:00
```

### Step 2 — Set up the environment

From the project directory, source the setup script (do not execute it — it must set
variables in your current shell):

```bash
cd /global/u2/r/rebassoo/work/fme/e3sm-ai-configs/ACE-OlawaleSampleTraining-E3SM-ACE
source setup-interactive-env.sh
```

This stages the config, sets all required env vars, and prints the `srun` command to use.

### Step 3 — Launch

```bash
srun -u $CONFIG_DIR/requeueable-train.sh
```

### Interactive shell inside the container

To get a bash shell inside the container on a single node (useful for debugging):

```bash
podman-hpc run --rm --gpu --nccl --ipc host --network=host \
  -v "${CONFIG_DIR}:/configmount" \
  -v "${FME_OUTPUT_DIR}:/output" \
  -v "${FME_TRAIN_DIR}:/traindata" \
  -v "${FME_VALID_DIR}:/validdata" \
  -v "${FME_STATS_DIR}:/statsdata" \
  -it $FME_IMAGE bash
```

(Requires `source setup-interactive-env.sh` first so the volume paths are set.)
