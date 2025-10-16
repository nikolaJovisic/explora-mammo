import logging
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import random
from torchvision.transforms.functional import crop, resize
from torch.utils.data import Dataset
import cv2
from PIL import ImageFilter
import torchvision.transforms.functional as F

logger = logging.getLogger("dinov2")


class RandomGammaCorrection:
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return F.adjust_gamma(img, gamma)
        return img

class RandomGaussianBlur:
    def __init__(self, sigma=(0.1, 0.3), p=0.3):
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img
        
class RandomGaussianNoise:
    def __init__(self, std_range=(0.01, 0.03), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img).astype(np.float32) / 255.0
            std = random.uniform(*self.std_range)
            # Create single-channel noise and broadcast it across all channels
            noise = np.random.normal(0, std, img_np.shape[:2])
            noise = np.repeat(noise[:, :, np.newaxis], img_np.shape[2], axis=2)
            noisy = np.clip(img_np + noise, 0, 1)
            return Image.fromarray((noisy * 255).astype(np.uint8))
        return img

class RandomElasticDeformation:
    def __init__(self, alpha=(5, 15), sigma=(3, 6), grid_size=(16, 32), p=1.0):
        self.alpha = alpha
        self.sigma = sigma
        self.grid_size = grid_size
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        # sample random parameters
        alpha = random.uniform(*self.alpha)
        sigma = random.uniform(*self.sigma)
        grid_size = int(random.uniform(*self.grid_size))

        img_np = np.array(img)
        shape = img_np.shape[:2]

        # downsample displacement field
        small_shape = (shape[0] // grid_size, shape[1] // grid_size)
        dx_small = np.random.rand(*small_shape) * 2 - 1
        dy_small = np.random.rand(*small_shape) * 2 - 1

        # upsample to full resolution and blur
        dx = cv2.resize(dx_small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy_small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        warped = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(warped)

class GrayscaleToRGBTransform:
    def __call__(self, img):
        img_np = np.array(img)
        if img_np.ndim == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        return Image.fromarray(img_rgb)

class RandomRightAngleRotation:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return img.rotate(angle)

class CustomRandomCrop:
    def __init__(self, size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC, threshold=0.2, max_tries=50):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.interpolation = interpolation
        self.threshold = threshold
        self.max_tries = max_tries

    def __call__(self, img):
        img_w, img_h = img.size
        min_dim = min(img_w, img_h)
        base_area = min_dim ** 2

        for _ in range(self.max_tries):
            target_area = base_area * torch.empty(1).uniform_(*self.scale).item()
            crop_size = int(round(target_area ** 0.5))

            if crop_size <= min_dim:
                if img_w > img_h:
                    top = torch.randint(0, img_h - crop_size + 1, (1,)).item()
                    left = torch.randint(0, img_w - crop_size + 1, (1,)).item()
                else:
                    top = torch.randint(0, img_h - crop_size + 1, (1,)).item()
                    left = torch.randint(0, img_w - crop_size + 1, (1,)).item()

                cropped = crop(img, top, left, crop_size, crop_size)
                resized = resize(cropped, self.size, interpolation=self.interpolation)

                if np.mean(np.array(resized)) / 255 > self.threshold:
                    return resized

        return resize(img, self.size, interpolation=self.interpolation)

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=518,
        local_crops_size=98,
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")


        self.intensity_augmentation = transforms.Compose([
            RandomGammaCorrection(p=0.5),
            RandomGaussianBlur(p=0.3),
            RandomGaussianNoise(p=0.3),
        ])


        self.geometric_augmentation_global = transforms.Compose([
            CustomRandomCrop(global_crops_size, scale=global_crops_scale),
            transforms.RandomRotation(degrees=15),
            RandomElasticDeformation(alpha=(1, 2), sigma=(6, 10), grid_size=(48, 96), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        self.geometric_augmentation_local = transforms.Compose([
            CustomRandomCrop(local_crops_size, scale=local_crops_scale),
            RandomRightAngleRotation(),
            RandomElasticDeformation(alpha=(0.5, 1.5), sigma=(2, 4), grid_size=(8, 24), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])


        global_transfo2_extra = transforms.Compose(
            [
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*normalization),
            ]
        )

        self.to_rgb = GrayscaleToRGBTransform()

        self.global_transfo1 = transforms.Compose([self.to_rgb, self.intensity_augmentation, self.normalize])
        self.global_transfo2 = transforms.Compose([self.to_rgb, self.intensity_augmentation, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([self.to_rgb, self.intensity_augmentation, self.normalize])

    def __call__(self, images):
        output = {}

        im1_base = self.geometric_augmentation_global(random.choice(images))
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(random.choice(images))
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(random.choice(images))) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output, None



class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return self.transform(x)