GPUS=1
NNODES=1
NODE_RANK=${RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}   #3. localhost问题留意地方

NCCL_IB_HCA=`ibdev2netdev|awk '{print$1}'`
roce_PORT=":1"
NCCL_IB_HCA=${NCCL_IB_HCA}${roce_PORT}
NCCL_DEBUG=TRACE
OMPI_MCA_btl_tcp_if_include=eth0
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA
# export NCCL_DEBUG
# export OMPI_MCA_btl_tcp_if_include
# export NCCL_SOCKET_IFNAME
# export NCCL_IB_DISABLE
export NCCL_IB_GID_INDEX

export OMPI_MCA_btl_tcp_if_include=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=29400 --node_rank=$NODE_RANK --nproc_per_node=$GPUS trainer.py --model_name pythia-14m --config config_hub/pretrain/debug_new.yaml
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=29400 --node_rank=$NODE_RANK --nproc_per_node=$GPUS trainer.py --config config_hub/pretrain/debug_infini_moe_mod.yaml
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=29500 --node_rank=$NODE_RANK --nproc_per_node=$GPUS trainer.py --config config_hub/pretrain/debug_infini.yaml

CUDA_VISIBLE_DEVICES=0 python inference_base.py --prompt "Once upon a time, " --checkpoint_dir "/home/notebook/code/personal/80234819/llm/litgpt-main/out/pretrain/debug_mamba_1.1"
