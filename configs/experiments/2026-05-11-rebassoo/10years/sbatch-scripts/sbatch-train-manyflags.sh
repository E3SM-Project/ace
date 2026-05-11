#!/bin/bash -l

#SBATCH -A m4492_g
##SBATCH -A e3sm_g
#SBATCH -q premium
#SBATCH -C gpu_hbm40g
#SBATCH -J train-fme
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 23:00:00
#SBATCH --output=joblogs/%j.out
#SBATCH --signal=USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

# for pre-emptible jobs, update -q to preempt

set -xe

# directory for saving output from training/inference job
if [ -z "${RESUME_JOB_ID}" ]; then
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${SLURM_JOB_ID}
else
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${RESUME_JOB_ID}
fi
mkdir -p $FME_OUTPUT_DIR

# Your existing Slingshot workarounds
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd

# Increase CXI buffer sizes to prevent exhaustion
export FI_CXI_OFLOW_BUF_SIZE=16777216      # 16MB (up from 8MB)
export FI_CXI_REQ_BUF_SIZE=16777216        # 16MB request buffer
export FI_CXI_RX_MATCH_MODE=hybrid         # Hybrid can be more stable than software

# Critical for multi-node on Slingshot
export NCCL_NET_GDR_LEVEL=PHB           # PCIe-Host-Bridge only (no GPU Direct RDMA)
export NCCL_CROSS_NIC=1                 # Allow cross-NIC communication
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3  # All 4 Slingshot NICs

# Disable problematic NCCL 2.26 features
export NCCL_NVLS_ENABLE=0               # No NVLink Sharp
export NCCL_CUMEM_ENABLE=0              # No CUDA unified memory
export NCCL_DMABUF_ENABLE=0             # Disable DMA-BUF
export NCCL_GRAPH_REGISTER=0

# Force more conservative algorithms for multi-node
export NCCL_ALGO=Tree                   # Tree topology instead of Ring
export NCCL_PROTO=Simple                # Simple protocol (no LL/LL128)
export NCCL_P2P_NET_CHUNKSIZE=131072    # 128KB chunks
export NCCL_MIN_NCHANNELS=4             # Minimum 4 channels
export NCCL_MAX_NCHANNELS=8            # Maximum 16 channels

# Prevent buffer reuse issues
export NCCL_BUFFSIZE=4194304               # 4MB buffer (larger for stability)

# Timeouts
export NCCL_TIMEOUT=1800000

# Debug (you can reduce verbosity once stable)
export NCCL_DEBUG=WARN                     # Less spam than INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000

export head_node=$(hostname)
export NCCL_DEBUG=INFO
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM

# TODO: automate

#fme_venv=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
#module load python
#conda activate $fme_venv

# env variables
export WANDB_JOB_TYPE=training
set +x  # don't print API key to logs
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

# run the requeueable job
srun -u --mpi=pmi2 $CONFIG_DIR/requeueable-train.sh

sleep 120
