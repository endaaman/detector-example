import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

np.random.seed(42)


def pad_targets(bboxes, labels, size, fill=(0, 0)):
    pad_bboxes = np.zeros([size, 4], dtype=bboxes.dtype)
    pad_labels = np.zeros([size], dtype=bboxes.dtype)
    if fill:
        pad_bboxes.fill(fill[0])
        pad_labels.fill(fill[1])
    pad_bboxes[0:bboxes.shape[0], :] = bboxes
    pad_labels[0:labels.shape[0]] = labels
    return pad_bboxes, pad_labels


def xyxy_to_xywh(bb, w, h):
    ww = (bb[:, 2] - bb[:, 0])
    hh = (bb[:, 3] - bb[:, 1])
    xx = bb[:, 0] + ww * 0.5
    yy = bb[:, 1] + hh * 0.5
    bb = np.stack([xx, yy, ww, hh], axis=1)
    bb[:, [0, 2]] /= w
    bb[:, [1, 3]] /= h
    return bb

class CircleDataset(Dataset):
    def __init__(self, target, image_size=512, bg=(0, 0, 0), fg=(255, 0, 0), normalized=True):
        adapter_table = {
            'effdet': self.effdet_adapter,
            'yolo': self.yolo_adapter,
            'ssd': self.ssd_adapter,
        }
        if target not in list(adapter_table.keys()):
            raise ValueError(f'Invalid target: {target}')
        self.bg = bg
        self.fg = fg
        self.image_size = image_size
        self.adapter = adapter_table[target]

        # 適当なaugmentaion
        self.albu = A.Compose([
            A.RandomResizedCrop(width=self.image_size, height=self.image_size, scale=[0.8, 1.0]),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            # 可視化するとき正規化されるとnoisyなのでトグれるようにする
            A.Normalize(mean=[0.2, 0.1, 0.1], std=[0.2, 0.1, 0.1]) if normalized else None,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return 1000 # 1epochあたりの枚数。自動生成なので適当

    def __getitem__(self, idx):
        img = Image.new('RGB', (512, 512), self.bg)

        size = np.random.randint(10, 256)
        left = np.random.randint(0, 256)
        top = np.random.randint(0, 256)

        right = left + size
        bottom = top + size
        draw = ImageDraw.Draw(img)
        draw.ellipse((left, top, right, bottom), fill=self.fg)

        # shapeはbox_count x box_coords (N x 4)。円は常に一つなので、今回は画像一枚に対して(1 x 4)
        bboxes = np.array([
            # albumentationsにはASCAL VOC形式の[x0, y0, x1, y1]をピクセル単位で入力する
            [left, top, right, bottom,],
        ])

        labels = np.array([
            # 検出対象はid>=1である必要あり。0はラベルなしとして無視される。
            1,
        ])

        result = self.albu(
            image=np.array(img),
            bboxes=bboxes,
            labels=labels,
        )
        images = result['image']
        bboxes = np.array(result['bboxes'])
        labels = np.array(result['labels'])
        return self.adapter(images, bboxes, labels)


    def effdet_adapter(self, images, bboxes, labels):
        # use yxyx
        bboxes = bboxes[:, [1, 0, 3, 2]]
        bboxes, labels = pad_targets(bboxes, labels, 1, fill=(0, -1))
        labels = {
            'bbox': torch.FloatTensor(bboxes),
            'cls': torch.FloatTensor(labels),
        }
        return images, labels

    def yolo_adapter(self, images, bboxes, labels):
        bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        bboxes, labels = pad_targets(bboxes, labels, 1, fill=(0, -1))
        batch_index = np.zeros([1, 1])
        # yolo targets: [batch_idx, class_id, x, y, w, h]
        labels = np.concatenate([batch_index, labels[:, None], bboxes], axis=1)
        return images, torch.FloatTensor(labels)

    def ssd_adapter(self, images, bboxes, labels):
        # bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        bboxes[:, [0, 2]] /= images.shape[2]
        bboxes[:, [1, 3]] /= images.shape[1]
        bboxes = [b for b in bboxes]
        labels = [l for l in labels]
        return images, (bboxes, labels)

if __name__ == '__main__':
    # draw_bounding_boxesはxyxy形式
    ds = CircleDataset(use_yxyx=False, normalized=False)
    for (x, y) in ds:
        to_pil_image(x).save(f'example_x.png')
        t = draw_bounding_boxes(image=x, boxes=y['bbox'], labels=[str(v.item()) for v in y['cls']])
        img = to_pil_image(t)
        img.save(f'example_xy.png')
        break
