import numpy as np
import torch
from . import SSD300
from .ssd import MultiBoxLoss

if __name__ == '__main__':
    model = SSD300(6)
    t = torch.ones([2, 3, 300, 300]).type(torch.FloatTensor)
    predicted_locs, predicted_scores = model(t)  # (N, 8732, 4), (N, 8732, n_classes)

    print(predicted_locs.shape)
    print(predicted_scores.shape)

    # labels = torch.zeros([2, 8732])
    # boxes = torch.zeros([2, 8732, 4])

    labels = torch.ones([2, 19])
    boxes = torch.zeros([2, 19, 4])

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
    loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
    print(loss)
