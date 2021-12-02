# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/8 11:27

import os
import cv2 as cv
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, data, image_root="", transform=None, train=True):
        super().__init__()
        self.data = data
        self.transform = transform
        self.image_root = image_root
        self.train = train
        self.feature_map_size = [37, 18, 9, 5, 3, 1]
        self.default_boxes_num = [4, 6, 6, 6, 4, 4]
        self.Sk = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_dict = self.data[index]
        folder = image_dict["folder"]
        name = image_dict["filename"]
        image_path = os.path.join(self.image_root, *folder, "JPEGImages", name)
        img = cv.imread(image_path)
        category = image_dict["category"]
        bbox = image_dict["bndbox"] # shape: nbox, 4
        if self.train: # 数据增强
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hsv_img = self.hue(hsv_img)
            hsv_img = self.saturation(hsv_img)
            hsv_img = self.value(hsv_img)
            img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
            img = self.average_blur(img)
            img, bbox = self.horizontal_flip(img, bbox)
            img, bbox, category = self.crop(img, bbox, category)
            img, bbox = self.scale(img, bbox)
            img, bbox, category = self.translation(img, bbox, category)
        # Caused by data augmentation, such as crop, translation and so on.
        if len(bbox) == 0 or len(category) == 0:
            return [], []
        target = self.encoder(img, bbox, category)
        img = cv.resize(img, (300, 300))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, target

    def hue(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            enhanced_hue = hsv_img[:, :, 0] * factor
            enhanced_hue = np.clip(enhanced_hue, 0, 180).astype(hsv_img.dtype)  # H的范围是0-180(360/2)
            hsv_img[:, :, 0] = enhanced_hue
        return hsv_img

    def saturation(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            enhanced_saturation = hsv_img[:, :, 1] * factor
            enhanced_saturation = np.clip(enhanced_saturation, 0, 255).astype(hsv_img.dtype)
            hsv_img[:, :, 1] = enhanced_saturation
        return hsv_img

    def value(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            enhanced_value = hsv_img[:, :, 2] * factor
            enhanced_value = np.clip(enhanced_value, 0, 255).astype(hsv_img.dtype)
            hsv_img[:, :, 2] = enhanced_value
        return hsv_img

    def average_blur(self, img):
        if random.random()>0.5:
            img = cv.blur(img, (3, 3))
        return img

    def horizontal_flip(self, img, bbox):
        if random.random() > 0.5:
            img = cv.flip(img, 1)
            h, w, _ = img.shape
            temp = w - bbox[:, 0]
            bbox[:, 0] = w - bbox[:, 2]
            bbox[:, 2] = temp
        return img, bbox

    def crop(self, img, bbox, category):
        if random.random() > 0.5:
            factor_horizontal = random.uniform(0, 0.2)
            factor_vertical = random.uniform(0, 0.2)
            h, w, _ = img.shape
            start_horizontal = int(w * factor_horizontal)
            end_horizontal = start_horizontal + int(0.8 * w)
            start_vertical = int(h * factor_vertical)
            end_vertical = start_vertical + int(0.8 * h)
            img = img[start_vertical: end_vertical, start_horizontal:end_horizontal, :]
            center_x = (bbox[:, 0] + bbox[:, 2]) / 2
            center_y = (bbox[:, 1] + bbox[:, 3]) / 2
            inImage = (center_x > start_horizontal) & (center_x < end_horizontal) \
                      & (center_y > start_vertical) & (center_y < end_vertical)
            bbox = bbox[inImage, :]
            bbox[:, [0, 2]] = bbox[:, [0, 2]] - start_horizontal
            bbox[:, [1, 3]] = bbox[:, [1, 3]] - start_vertical
            bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, int(0.8 * w))
            bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, int(0.8 * h))
            category = category[inImage]
        return img, bbox, category

    def scale(self, img, bbox):
        probility = random.random()
        if probility > 0.7:
            factor = random.uniform(0.5, 1.5)
            h, w, _ = img.shape
            h = int(h * factor)
            img = cv.resize(img, (w, h))  # size的顺序是w,h
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * factor
        elif probility < 0.3:
            factor = random.uniform(0.5, 1.5)
            h, w, _ = img.shape
            w = int(w * factor)
            img = cv.resize(img, (w, h))
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * factor
        bbox = bbox.astype(np.int)
        return img, bbox

    def translation(self, img, bbox, category):
        if random.random() > 0.5:
            factor_horizontal = random.uniform(-0.2, 0.2)
            factor_vertical = random.uniform(-0.2, 0.2)
            h, w, _ = img.shape
            w_tran = int(w * factor_horizontal)
            h_tran = int(h * factor_vertical)
            canvas = np.zeros_like(img)
            if w_tran < 0 and h_tran < 0:  # 向右下移动
                canvas[-h_tran:, -w_tran:, :] = img[:h + h_tran, :w + w_tran, :]
            elif w_tran < 0 and h_tran >= 0:  # 向右上移动
                canvas[:h - h_tran, -w_tran:, :] = img[h_tran:, :w + w_tran, :]
            elif w_tran >= 0 and h_tran < 0:  # 向左下移动
                canvas[-h_tran:, :w - w_tran, :] = img[:h + h_tran, w_tran:, :]
            elif w_tran >= 0 and h_tran >= 0:  # 向左上移动
                canvas[:h - h_tran, :w - w_tran, :] = img[h_tran:, w_tran:, :]
            bbox[:, [0, 2]] = bbox[:, [0, 2]] - w_tran
            bbox[:, [1, 3]] = bbox[:, [1, 3]] - h_tran
            # 确保bbox中心点在图像内，因为中心点所在的格负责预测
            center_x = (bbox[:, 0] + bbox[:, 2]) / 2  # shape: nbox
            center_y = (bbox[:, 1] + bbox[:, 3]) / 2  # shape: nbox
            inImage = ((center_x > 0) & (center_x < w)) & ((center_y > 0) & (center_y < h))
            bbox = bbox[inImage, :]
            bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, w)  # 中心虽然还在图片内，但是边框可能会超过边界，要限制范围
            bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, h)
            category = category[inImage]
            return canvas, bbox, category
        return img, bbox, category

    def encoder(self, img, bbox, category):
        bbox = torch.from_numpy(bbox)
        category = torch.from_numpy(category)
        h, w, _ = img.shape
        w_scale_factor = 300 / w
        h_scale_factor = 300 / h
        bbox[:, [0, 2]] = (bbox[:, [0, 2]] * w_scale_factor).long()
        bbox[:, [1, 3]] = (bbox[:, [1, 3]] * h_scale_factor).long()
        # 1、确保每个gt都能匹配到至少一个default box (解决：多个gt匹配到同一个default box的情况)
        # 2、把多个default box分配到同一个gt
        iou, default_box = self.calculate_iou(bbox, self.feature_map_size, self.default_boxes_num, self.Sk) # n, 8096
        _, selected_default_box_idx = torch.max(iou, dim=1, keepdim=True)  # 保证每个gt都能有对应的default box
        selected_gt_iou, selected_gt_idx = torch.max(iou, dim=0, keepdim=True)  # 每个default box选择一个gt，实现多个default box对应一个gt
        selected_default_box_idx = selected_default_box_idx.squeeze(-1)
        selected_gt_iou = selected_gt_iou.squeeze(0)
        selected_gt_idx = selected_gt_idx.squeeze(0)
        selected_gt_iou[selected_default_box_idx] = 2 # 如果gt与对应的唯一default box的iou<0.5，这样可以保证不会在后面的操作中被标记为背景
        selected_gt_idx[selected_default_box_idx] = torch.arange(0, len(selected_default_box_idx))

        label = category[selected_gt_idx] + 1 # 每个default box对应的label
        mask = selected_gt_iou < 0.5
        label[mask] = 0 # 背景类
        label = label.unsqueeze(-1)
        coordinate = bbox[selected_gt_idx, :] # 8096, 4
        center_x = (coordinate[:, 0] + coordinate[:, 2]) // 2
        center_y = (coordinate[:, 1] + coordinate[:, 3]) // 2
        width = coordinate[:, 2] - coordinate[:, 0]
        height = coordinate[:, 3] - coordinate[:, 1]
        default_center_x = default_box[0]
        default_center_y = default_box[1]
        default_width = default_box[2]
        default_height = default_box[3]
        tx = (center_x - default_center_x)/default_width # 8096,
        tx = tx.unsqueeze(-1)
        ty = (center_y - default_center_y)/default_height
        ty = ty.unsqueeze(-1)
        tw = torch.log(width / default_width)
        tw = tw.unsqueeze(-1)
        th = torch.log(height / default_height)
        th = th.unsqueeze(-1)
        return torch.cat([tx, ty, tw, th, label], dim=-1) # # 8096, 5


    def calculate_iou(self, bbox, feature_map_size, default_box_num, Sk):
        # 逐层feature map计算
        iou_list = []
        center_x_list = []
        center_y_list = []
        width_list = []
        height_list = []
        gt_num = bbox.shape[0]
        for idx, (fms, dbn, sk) in enumerate(zip(feature_map_size, default_box_num, Sk)):
            center = torch.linspace(0.5, fms-0.5, fms, dtype=torch.float) / fms * 300 # fms,
            center_x = center.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # 1, 1, fms, 1
            center_x = center_x.repeat(1, fms, 1, dbn) # 1, fms, fms, dbn

            center_y = center.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # 1, fms, 1, 1
            center_y = center_y.repeat(1, 1, fms, dbn) # 1, fms, fms, dbn

            ratio = torch.Tensor([1, 2, 1 / 2, 3, 1 / 3])[:dbn-1]
            width_prime = height_prime = torch.Tensor([np.sqrt(sk * Sk[idx + 1]) * 300])

            width = sk * 300 / ratio
            width = torch.cat([width, width_prime], dim=0)  # dbn,
            width = width.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1, dbn
            width = width.repeat(1, fms, fms, 1) # 1, fms, fms, dbn

            height = sk * 300 * ratio
            height = torch.cat([height, height_prime], dim=0)  # dbn,
            height = height.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1, dbn
            height = height.repeat(1, fms, fms, 1) # 1, fms, fms, dbn

            x1 = center_x - width / 2 # 1, fms, fms, dbn
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2

            bbox_copy = bbox.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()  # n, 1, 1, 1, 4
            left = torch.max(bbox_copy[:, :, :, :, 0], x1)
            top = torch.max(bbox_copy[:, :, :, :, 1], y1)
            right = torch.min(bbox_copy[:, :, :, :, 2], x2)
            bottom = torch.min(bbox_copy[:, :, :, :, 3], y2)
            w = torch.max(right - left, torch.Tensor([1e-6]))
            h = torch.max(bottom - top, torch.Tensor([1e-6]))
            intersection = h * w
            union = (bbox_copy[:, :, :, :, 2] - bbox_copy[:, :, :, :, 0]) * (bbox_copy[:, :, :, :, 3] - bbox_copy[:, :, :, :, 1]) \
                    + (x2 - x1) * (y2 - y1) - intersection
            iou = intersection / union  # n, fms, fms, dbn
            iou_list.append(iou.reshape(gt_num, -1))
            center_x_list.append(center_x.reshape(-1))
            center_y_list.append(center_y.reshape(-1))
            width_list.append(width.reshape(-1))
            height_list.append(height.reshape(-1))
        return torch.cat(iou_list, dim=-1), (torch.cat(center_x_list, dim=0), torch.cat(center_y_list, dim=0),
                                             torch.cat(width_list, dim=0), torch.cat(height_list, dim=0))
# #
# if __name__ == "__main__":
#     import glob
#     import xml.dom.minidom as xdm
#     def load_data(data_path):
#         voc2007_trainval_annotations = os.path.join(data_path, "VOC2007", "trainval", "Annotations", "*xml")
#         annotation_path = glob.glob(voc2007_trainval_annotations)
#
#         all_category = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#                         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
#                         "train", "tvmonitor"]
#
#         random.shuffle(annotation_path)
#         res = []
#         for i, path in enumerate(annotation_path):
#             path_split = os.path.dirname(path).split('\\')
#             folder = [path_split[2], path_split[3]]  # [VOC2007/VOC2012, trainval/test]
#             DOMTree = xdm.parse(path)
#             collection = DOMTree.documentElement
#             filename = collection.getElementsByTagName("filename")[0].childNodes[0].data
#             category_list = []
#             bndbox_list = []
#             object_ = collection.getElementsByTagName("object")
#             for obj in object_:
#                 category = obj.getElementsByTagName("name")[0].childNodes[0].data
#                 bndbox = obj.getElementsByTagName("bndbox")[0]
#                 xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
#                 ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
#                 xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
#                 ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
#                 category_list.append(all_category.index(category))
#                 bndbox_list.append([xmin, ymin, xmax, ymax])
#             res.append({"folder": folder, "filename": filename, "category": np.array(category_list),
#                         "bndbox": np.array(bndbox_list)})
#         return res
#
#     DATA_PATH = "D:\\dataset"
#     res = load_data(DATA_PATH)
#     eval_res = res[:100]
#     eval_set = VOCDataset(data=eval_res, image_root=DATA_PATH, transform=None, train=False)
#     for data in eval_set:
#         print(data)