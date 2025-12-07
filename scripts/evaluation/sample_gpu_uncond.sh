export http_proxy=httpproxy.glm.ai:8888
export https_proxy=httpproxy.glm.ai:8888
export TORCH_HOME=/workspace/cogview_dev/xutd/xu/models
export WANDB_API_KEY=589d531f376ac4cfc8d713753b96052a1420b709

CHUNKS=8

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python sample.py \
        --ckpt "/workspace/cogview_dev/xutd/checkpoints/bsq_ffhq/test/epoch=299-step=41100.ckpt" \
        --o "/workspace/cogview_dev/xutd/npz/bsq_ffhq" \
        --config "./configs/IBQ/gpu/imagenet_conditional_llama_B_bsq_ffhq.yaml" \
        -k 0 \
        -p 1.0 \
        -n 50 \
        -t 1.15 \
        --batch_size 256 \
        --cfg_scale 1.0 \
        --model IBQ \
        --global_seed 42 \
        --num_chunks $CHUNKS \
        --num_samples 8750 \
        --classes "0" \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir /workspace/cogview_dev/xutd/npz/bsq_ffhq/samples/top_k_0_temp_1.15_top_p_1.0_cfg_1.0/epoch=299-step=41100.ckpt
