import torch

def masked_avgpool(sent, mask):
    mask_ = mask.masked_fill(mask == 0, -1e9).float()
    score = torch.softmax(mask_, -1)
    return torch.matmul(score.unsqueeze(1), sent).squeeze(1)