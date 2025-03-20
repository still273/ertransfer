from torch.nn import CosineSimilarity
import torch

def cos_sim(a,b):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    a.to(device)
    b.to(device)
    CS = CosineSimilarity(dim=1).to(device)
    similarity = CS(a,b)
    return similarity.detach().cpu()
