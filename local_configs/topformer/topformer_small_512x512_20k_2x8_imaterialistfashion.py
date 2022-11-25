_base_ = [
    "../_base_/datasets/imaterialist_fashion.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
    "./topformer_small_46classes.py",
]

num_gpus = 4
samples_per_gpu = 8

# By default, models are trained on 8 GPUs with 2 images per GPU.
# Use linear scaling rule to compensate in case of deviations
lr = 0.00012 * num_gpus * samples_per_gpu / (2 * 8)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)


data = dict(samples_per_gpu=samples_per_gpu, workers_per_gpu=samples_per_gpu)
find_unused_parameters = True
