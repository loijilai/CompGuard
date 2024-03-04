CUDA_VISIBLE_DEVICES=3 python train.py \
    --model bmshj2018-factorized \
    --dataset /tmp2/loijilai/compguard/dataset \
    --ckpt_outdir /tmp2/loijilai/compguard/checkpoints/factorized \
    --save \
    --epochs 3 \
    --pretrained

# Model
    * --model
    * --quality
    * --pretrained
    * --checkpoint
# Dataset
    * --dataset
    * --num_workers
    * --batch_size
    * --test_batch_size
    * --patch_size
# Training hyperparameters
    * --epochs
    * --learning_rate
    * --aux_learning_rate
    * --lambda
    * --clip_max_norm
# Environment
    * --save
    * --ckpt_outdir
    * --seed