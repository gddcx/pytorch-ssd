# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/11/28 14:18

import torch
import torch.nn as nn

class SSDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="sum")
        self.l1 = nn.SmoothL1Loss(reduction="sum")

    # out: bs, 8096, 25; target: bs, 8096, 5
    def forward(self, out, target):
        foreground = target[:, :, -1] > 0
        N = (foreground).sum().float()
        loc = self.l1(out[foreground, :][:, :4], target[foreground, :][:, :4])
        conf = self.ce(out[:, :, 4:].reshape(-1, 21), target[:, :, -1].reshape(-1).long())
        return (loc + conf) / N
