OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="6" \
  python change_detection/train.py \
    --backbone mixformer \
    --dataset cdd \
    --pretrained outputs/scene_recognition/mixformer_tiny_224/exp3/model_best.pth.tar \
    --exp tiny-exp1