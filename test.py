import torch
import segmentation_models_pytorch as smp
import dataset as ds
import utils as ut


if __name__ == '__main__':
    # load best saved checkpoint
    best_model = torch.load('./best_model.pth')

    # create test dataset
    test_dataset = ds.Dataset(
        ds.x_test_dir,
        ds.y_test_dir,
        augmentation=ut.get_validation_augmentation(),
        preprocessing=ut.get_preprocessing(ut.preprocessing_fn),
        classes=ut.CLASSES,
    )

    test_dataloader = ds.DataLoader(test_dataset)

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=ut.loss,
        metrics=ut.metrics,
        device=ut.DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

