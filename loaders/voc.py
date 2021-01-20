import os
import numpy as np
import cv2
import utils as ut
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """Pascal VOC Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace("jpg", "png")) for image_id in self.ids]

        # convert str names to class values on masks
        if classes is not None and len(classes) < len(self.CLASSES):
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = 'data/VOC2012/'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

# Lets look at data we have

# dataset = Dataset(x_train_dir, y_train_dir, classes=ut.CLASSES)
# image, mask = dataset[0]  # get some sample
# ut.visualize(
#     image=image,
#     mask=mask,
# )

# Visualize resulted augmented images and masks
# augmented_dataset = Dataset(
#     x_train_dir,
#     y_train_dir,
#     augmentation=ut.get_training_augmentation(),
#     classes=ut.CLASSES,
# )
#
# # same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[83]
#     ut.visualize(image=image, mask=mask)
# input()

# Input image transform normalize
# transform = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.Resize(img_size, interpolation=Image.BICUBIC),
#         torchvision.transforms.CenterCrop(img_size),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]
# )
