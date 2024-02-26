_base_ = [
    '../_base_/models/upernet_mixformer.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='outputs/scene_recognition/mixformer_tiny_224/exp3/model_best.pth.tar',
    backbone=dict(),
    decode_head=dict(num_classes=5, ignore_index=5), 
    auxiliary_head=dict(num_classes=5, ignore_index=5)
    )

optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.01,)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)