import sys
import torch
import numpy as np
import segmentation_models_pytorch as smp
import loaders.voc as ds
import utils as ut
import log_tb as log
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ut.ENCODER,
        encoder_weights=ut.ENCODER_WEIGHTS,
        classes=len(ut.CLASSES),
        activation=ut.ACTIVATION,
    )

    # model = torch.load("models/best_model.pth")

    train_dataset = ds.Dataset(
        ds.x_train_dir,
        ds.y_train_dir,
        augmentation=ut.get_training_augmentation(),
        preprocessing=ut.get_preprocessing(ut.preprocessing_fn),
        classes=ut.CLASSES,
    )

    valid_dataset = ds.Dataset(
        ds.x_valid_dir,
        ds.y_valid_dir,
        augmentation=ut.get_validation_augmentation(),
        preprocessing=ut.get_preprocessing(ut.preprocessing_fn),
        classes=ut.CLASSES,
    )

    # test dataset without transformations for image visualization
    valid_dataset_vis = ds.Dataset(
        ds.x_valid_dir, ds.y_valid_dir,
        classes=ut.CLASSES,
        augmentation=ut.get_validation_augmentation(),
    )

    if len(sys.argv) > 2:
        train_batch = int(sys.argv[1])
        valid_batch = int(sys.argv[2])
    else:
        train_batch = ut.DEF_TRN_BATCH
        valid_batch = ut.DEF_VAL_BATCH

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=False, num_workers=12)

    # for _, data in enumerate(train_loader, 0):
    #     images, labels = data
    #     pass

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=ut.loss,
        metrics=ut.metrics,
        optimizer=optimizer,
        device=ut.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=ut.loss,
        metrics=ut.metrics,
        device=ut.DEVICE,
        verbose=True,
    )

    # init log
    n = 0
    image_vis = np.moveaxis(valid_dataset_vis[n][0].astype('uint8'), -1, 0)
    image, gt_mask = valid_dataset[n]
    gt_mask = ut.encode_map(np.argmax(gt_mask, 0))
    gt_mask = gt_mask // 3 * 2 + image_vis // 3
    x_tensor = torch.from_numpy(image).to(ut.DEVICE).unsqueeze(0)

    log.init("VOC_v2")
    log.add(0, images=(None, gt_mask, image_vis))

    # train model for some epochs
    max_score = 0

    for i in range(0, 40):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, 'models/best_model.pth')
            print('Model saved!')

        pr_mask = model.predict(x_tensor).squeeze(0)
        pr_mask = ut.encode_map(torch.max(pr_mask, 0)[1].cpu().numpy().round())
        pr_mask = pr_mask // 3 * 2 + image_vis // 3

        log.add(i, (train_logs['iou_score'], valid_logs['iou_score'],
                    train_logs['dice_loss'], valid_logs['dice_loss'],
                    optimizer.param_groups[0]['lr']),
                (pr_mask, ))
        log.save()

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
