import cv2
import numpy as np
import random
import math


class CephaloTransform:
    """
    Augmentation pipeline for cephalometric landmark detection.
    Applies spatial and photometric augmentations while keeping landmarks consistent.
    """

    def __init__(
        self,
        is_train=True,
        rotation=15.0,
        scale=(0.85, 1.15),
        translate=0.05,
        brightness=0.3,
        contrast=0.3,
        noise_std=0.02,
        blur_prob=0.3,
    ):
        self.is_train = is_train
        self.rotation = rotation
        self.scale = scale
        self.translate = translate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.blur_prob = blur_prob

    def __call__(self, img, landmarks, imgsz):
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        if not self.is_train:
            return img, landmarks

        img, landmarks = self._random_affine(img, landmarks, imgsz)
        img = self._photometric(img)

        return img, landmarks

    def _random_affine(self, img, landmarks, imgsz):
        angle = random.uniform(-self.rotation, self.rotation)
        scale = random.uniform(self.scale[0], self.scale[1])
        tx = random.uniform(-self.translate, self.translate) * imgsz
        ty = random.uniform(-self.translate, self.translate) * imgsz

        cx, cy = imgsz / 2, imgsz / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        img = cv2.warpAffine(img, M, (imgsz, imgsz), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

        ones = np.ones((len(landmarks), 1), dtype=np.float32)
        pts_h = np.hstack([landmarks, ones])  # (N, 3)
        transformed = (M @ pts_h.T).T  # (N, 2)

        # Clamp to image bounds
        transformed[:, 0] = np.clip(transformed[:, 0], 0, imgsz - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, imgsz - 1)

        return img, transformed.astype(np.float32)

    def _photometric(self, img):
        img = img.astype(np.float32)

        # Brightness
        delta = random.uniform(-self.brightness, self.brightness) * 255
        img = img + delta
        img = np.clip(img, 0, 255)

        # Contrast
        alpha = random.uniform(1 - self.contrast, 1 + self.contrast)
        mean_val = img.mean()
        img = mean_val + alpha * (img - mean_val)
        img = np.clip(img, 0, 255)

        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std * 255, img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)

        # Gaussian blur (occasional)
        if random.random() < self.blur_prob:
            ksize = random.choice([3, 5])
            img_uint8 = img.astype(np.uint8)
            img_uint8 = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
            img = img_uint8.astype(np.float32)

        return img.astype(np.uint8)


def build_transforms(is_train=True, imgsz=512, augment=True, **kwargs):
    if is_train and augment:
        return CephaloTransform(
            is_train=True,
            rotation=kwargs.get('rotation', 15.0),
            scale=kwargs.get('scale', (0.85, 1.15)),
            translate=kwargs.get('translate', 0.05),
            brightness=kwargs.get('brightness', 0.3),
            contrast=kwargs.get('contrast', 0.3),
            noise_std=kwargs.get('noise_std', 0.02),
            blur_prob=kwargs.get('blur_prob', 0.3),
        )
    return CephaloTransform(is_train=False)
