_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'


data_root = 'mmdetection/data/ewis/'
class_name = ('crop','weed')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='train_images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val_images/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test_images/')))

val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = dict(ann_file=data_root + 'test.json')

evaluation = dict(
    interval=1,  # Evaluate after every epoch
    metric='mAP',  # Use mAP as the evaluation metric
    save_best='coco/bbox_mAP',  # Save the best model based on mAP for bounding boxes
    rule='greater'  # Higher mAP values are better
)


max_epoch = 10


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', save_best='coco/bbox_mAP', rule='greater',interval = 1, max_keep_ckpts = 1, by_epoch = True),
    early_stopping = dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', patience=10, rule='greater'),
    )

train_cfg = dict(max_epochs=max_epoch, val_interval=1)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'Grounding DINO experiment',
            'group': 'grounding_dino_swin-t_finetune_16xb2_1x_coco'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
runner = dict(type='EpochBasedRunner', max_epochs=1)