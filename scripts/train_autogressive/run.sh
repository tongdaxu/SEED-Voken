# sed -i "171d" /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py > /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py

export http_proxy=httpproxy.glm.ai:8888
export https_proxy=httpproxy.glm.ai:8888
export TORCH_HOME=/workspace/cogview_dev/xutd/xu/models
export WANDB_API_KEY=589d531f376ac4cfc8d713753b96052a1420b709

export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10063}
export NODE_RANK=${3:-0}

export OMP_NUM_THREADS=6
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo $MASTER_ADDR
echo $MASTER_PORT

# GPU
NODE_RANK=$NODE_RANK python main.py fit --config configs/IBQ/gpu/imagenet_conditional_llama_B_bsq_ffhq.yaml
