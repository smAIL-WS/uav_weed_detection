_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = 'mmyolo/data/ewis/'
class_name = ('crop', 'weed')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (220, 20, 60)])

close_mosaic_epochs = 5

max_epochs = 300
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
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

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs



default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)


visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend',init_kwargs={
            'project': 'Yolov8 experiment',
            'group': 'yolov8_s_fast_1xb12-40e_coco'
         })])

