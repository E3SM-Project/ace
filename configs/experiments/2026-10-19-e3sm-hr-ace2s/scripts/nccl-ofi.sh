# Source me to put NERSC's aws-ofi-nccl (libfabric/CXI) plugin under torch's
# bundled NCCL. Without it NCCL falls back to TCP sockets over hsn.
P=/global/common/software/nersc9/nccl/2.24.3/plugin
export LD_LIBRARY_PATH=$P/lib:$P/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=PHB NCCL_SOCKET_IFNAME=hsn
export FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_RDZV_THRESHOLD=0
