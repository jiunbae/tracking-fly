import typing

import numpy as np
import torch
import torchvision

from flytracking.detection.models.efficientdet_base import EfficientDetBackbone
from flytracking.detection.utils.efficientdet import (
    BBoxTransform,
    ClipBoxes,
)
from flytracking.detection.utils.process import (
    aspectaware_resize_padding,
    invert_affine,
)


class Detector:
    SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    CLASSES = ['fly']
    ANCHOR_RATIO = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    ANCHOR_SCALE = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    def __init__(
        self, threshold: float = .15, iou_threshold: float = .15, compound_coef: int = 2,
        weights: str = 'weights/efficientdet-d2.pth', device: str = 'cuda',
    ):
        self.device = torch.device(device)
        self.compound_coef = compound_coef
        self.threshold = threshold
        self.iou_threshold = iou_threshold

        self.model = EfficientDetBackbone(
            compound_coef=compound_coef, num_classes=len(self.CLASSES),
            ratios=self.ANCHOR_RATIO, scales=self.ANCHOR_SCALE,
        )
        self.model.load_state_dict(torch.load(weights, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.input_size = self.SIZES[compound_coef]

    @staticmethod
    def preprocess(
        images: np.ndarray,
        max_size: int = 512,
        mean: typing.Tuple[float] = (0.485, 0.456, 0.406),
        std: typing.Tuple[float] = (0.229, 0.224, 0.225)
    ) -> typing.Tuple[np.ndarray, typing.List[typing.Tuple[float, ...]]]:
        normalized = (images[..., ::-1] / 255 - mean) / std
        img_metas = [
            aspectaware_resize_padding(img, max_size, max_size, means=None)
            for img in normalized
        ]

        images, *metas = zip(*img_metas)
        metas = list(zip(*metas))
        return images, metas

    def decode(
        self,
        tensor: torch.Tensor,
        anchors: torch.Tensor,
        regression: torch.Tensor,
        classification: torch.Tensor,
    ) -> typing.List[np.ndarray]:
        def decode_(
            classification_: torch.Tensor,
            transformed_anchors_: torch.Tensor,
            scores_per: torch.Tensor,
        ) -> np.ndarray:
            scores_, classes = classification_.max(dim=0)
            anchors_nms_idx = torchvision.ops.boxes.batched_nms(
                transformed_anchors_,
                scores_per[:, 0],
                classes,
                iou_threshold=self.iou_threshold
            )

            if anchors_nms_idx.shape[0] != 0:
                return torch.cat(
                    (
                        classes[anchors_nms_idx].unsqueeze(-1),
                        transformed_anchors_[anchors_nms_idx, :],
                        scores_[anchors_nms_idx].unsqueeze(-1),
                    ),
                    dim=-1,
                ).detach().cpu().numpy()

            return np.empty((0, 6))

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, tensor)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.threshold)[:, :, 0]
        batch_size = tensor.size(0)
        results = [
            np.empty((0, 6)) if (scores_over_thresh[i].sum() == 0) else
            decode_(
                classification_=classification[i, scores_over_thresh[i, :], ...].permute(1, 0),
                transformed_anchors_=transformed_anchors[i, scores_over_thresh[i, :], ...],
                scores_per=scores[i, scores_over_thresh[i, :], ...],
            ) for i in range(batch_size)
        ]

        return results

    @staticmethod
    def remove_class_ids(detections: typing.List[np.ndarray]) \
            -> typing.List[np.ndarray]:
        return [
            detection[:, 1:]
            for detection in detections
        ]

    def __call__(self, img):
        framed_images, framed_metas = self.preprocess(img, max_size=self.input_size)

        tensor = torch.stack([
            torch.from_numpy(image).to(self.device)
            for image in framed_images
        ], 0).to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            _, regression, classification, anchors = self.model(tensor)

            results = self.decode(
                tensor.detach().cpu(),
                anchors.detach().cpu(),
                regression.detach().cpu(),
                classification.detach().cpu(),
            )
            results = invert_affine(framed_metas, results)
            results = self.remove_class_ids(results)

            return results
