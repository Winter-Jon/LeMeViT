OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="7" \
python benchmark.py  --results-file benchmark.txt \
  --model lemevit_tiny \
  --bench both \
  --num-bench-iter 100 \
  --batch-size 256 --img-size 224 --num-classes 1000 \
  --opt adamw --opt-eps 1e-8 --momentum 0.9 --weight-decay 0.05 \
  --smoothing 0.1 --drop-path 0.1 \
  --amp --channels-last \