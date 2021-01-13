import torch
import numpy as np
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

    # test dataset without transformations for image visualization
    test_dataset_vis = ds.Dataset(
        ds.x_test_dir, ds.y_test_dir,
        classes=ut.CLASSES,
    )

    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(ut.DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        ds.visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )

