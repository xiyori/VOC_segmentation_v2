import torch
import segmentation_models_pytorch as smp
import loaders.voc as ds
import utils as ut
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # load best saved checkpoint
    best_model = torch.load('models/voc_model_70%_iou.pth')

    # create test dataset
    test_dataset = ds.Dataset(
        ds.x_test_dir,
        ds.y_test_dir,
        augmentation=ut.get_validation_augmentation(),
        preprocessing=ut.get_preprocessing(ut.preprocessing_fn),
        classes=ut.CLASSES,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=6)

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=ut.loss,
        metrics=ut.metrics,
        device=ut.DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

