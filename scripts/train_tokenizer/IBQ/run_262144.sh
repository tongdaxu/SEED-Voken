sed -i "171d" /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py > /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py
sed -i 's/import torchvision.transforms.functional_tensor as F_t/import torchvision.transforms._functional_tensor as F_t/g' /usr/local/lib/python3.10/dist-packages/pytorchvideo/transforms/augmentations.py

export http_proxy=httpproxy.glm.ai:8888
export https_proxy=httpproxy.glm.ai:8888
export TORCH_HOME=/workspace/cogview_dev/xutd/xu/models
export WANDB_API_KEY=589d531f376ac4cfc8d713753b96052a1420b709

export OMP_NUM_THREADS=6

NODE_RANK=0 python main.py fit --config configs/IBQ/gpu/imagenet_ibqgan_262144.yaml
