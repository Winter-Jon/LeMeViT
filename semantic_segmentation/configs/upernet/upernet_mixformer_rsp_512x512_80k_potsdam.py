_base_ = [
    '../_base_/models/upernet_mixformer.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='./outputs/scene_recognition/mixformer_tiny_224/exp6/model_best.pth.tar',
    backbone=dict(),
    decode_head=dict(num_classes=6,ignore_index=6), 
    auxiliary_head=dict(num_classes=6,ignore_index=6)
    )