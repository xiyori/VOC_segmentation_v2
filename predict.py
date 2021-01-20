import torch
import numpy as np
import loaders.voc as ds
import utils as ut


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

    # test dataset without transformations for image visualization
    test_dataset_vis = ds.Dataset(
        ds.x_test_dir, ds.y_test_dir,
        classes=ut.CLASSES,
        augmentation=ut.get_validation_augmentation(),
    )

    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = np.moveaxis(test_dataset_vis[n][0].astype('uint8'), -1, 0)
        image, gt_mask = test_dataset[n]

        gt_mask = ut.encode_map(np.argmax(gt_mask, 0))
        gt_mask = gt_mask // 3 * 2 + image_vis // 3
        gt_mask = np.moveaxis(gt_mask, 0, -1)

        x_tensor = torch.from_numpy(image).to(ut.DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor).squeeze(0)
        pr_mask = ut.encode_map(torch.max(pr_mask, 0)[1].cpu().numpy().round())
        pr_mask = pr_mask // 3 * 2 + image_vis // 3
        pr_mask = np.moveaxis(pr_mask, 0, -1)

        image_vis = np.moveaxis(image_vis, 0, -1)

        ut.visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )

