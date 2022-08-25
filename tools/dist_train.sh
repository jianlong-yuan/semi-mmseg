CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$RANDOM}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
WORK_DIR=${WORK_DIR:-"$(echo ${CONFIG%.*} | sed -e "s/projects\///g;s/.*configs/work_dirs/g")/"}
echo "you have input the path $WORK_DIR"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --work-dir $WORK_DIR \
    --launcher pytorch ${@:3}
