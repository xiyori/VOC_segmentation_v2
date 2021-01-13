import torch
import numpy as np
import segmentation_models_pytorch as smp
import dataset as ds
import utils as ut
import log_tensorboard as log


if __name__ == '__main__':
    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ut.ENCODER,
        encoder_weights=ut.ENCODER_WEIGHTS,
        classes=len(ut.CLASSES),
        activation=ut.ACTIVATION,
    )

    # model = torch.load("best_model.pth")

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
    test_dataset_vis = ds.Dataset(
        ds.x_test_dir, ds.y_test_dir,
        classes=ut.CLASSES,
    )

    train_loader = ds.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=12)
    valid_loader = ds.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

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
    image_vis = np.moveaxis(test_dataset_vis[n][0].astype('uint8'), -1, 0)
    image, gt_mask = valid_dataset[n]
    # gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(ut.DEVICE).unsqueeze(0)

    log.init("CamVid")
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
            torch.save(model, './best_model.pth')
            print('Model saved!')

        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze(0).cpu().numpy().round())

        log.add(i, (train_logs['iou_score'], valid_logs['iou_score'],
                    train_logs['dice_loss'], valid_logs['dice_loss'],
                    optimizer.param_groups[0]['lr']),
                (pr_mask, ))
        log.save()

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
