CUDA_VISIBLE_DEVICES=3 python inference.py \
    --model bmshj2018-factorized \
    --input /tmp2/loijilai/compguard/dataset/monalisa.jpg \
    --output /tmp2/loijilai/compguard/outputs/factorized/output.png \
    --recompress 1 \
    --patch_size 256 256\
    --checkpoint /tmp2/loijilai/compguard/checkpoints/factorized/checkpoint_pretrained_best_loss.pth.tar

# Model
    * --model
    * --quality
    * --checkpoint (if no checkpoint, then use pretrained model)
# Image
    * --input
    * --output
    * --patch_size
# Inference setting
    * --recompress