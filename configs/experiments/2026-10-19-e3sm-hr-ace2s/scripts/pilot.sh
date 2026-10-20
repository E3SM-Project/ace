#!/bin/bash
# Pilot: the production config on the nodes of the current allocation, with
# the recommended model (spectral modes x0.5) and loader settings measured on
# 2026-10-19. Resumable: rerunning with the same EXPERIMENT_DIR continues from
# training_checkpoints/ckpt.tar (e.g. later with scripts/sbatch-train-hr.sh,
# EXPDIR=<same dir> and CONFIG=<this run's config.yaml>).
#   srun -N <n> --ntasks-per-node 1 --gpus-per-node 4 scripts/pilot.sh DIR
here=$(cd "$(dirname "$0")" && pwd)
n=$SLURM_NNODES
exec "$here/train.sh" 4 1 "$here/../config-train-hr.yaml" "$1" \
  stepper.step.config.builder.config.hard_thresholding_fraction=0.5 \
  train_loader.batch_size="$n" validation.loader.batch_size="$n" \
  train_loader.num_data_workers=2 validation.loader.num_data_workers=2 \
  checkpoint_every_n_batches=200
