import torch
from torchvision.datasets import CocoDetection

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class Sidewalk(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(Sidewalk, self).__init__(root, annFile, transforms)
        self.transforms = transforms

    def __getitem__(self, index):
        img, anno = super(Sidewalk, self).__getitem__(index)

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index