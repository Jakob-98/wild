try:
    import torch
    torchAvailable = True
except ImportError:
    pass
class params:
    batch_size = 16
    img_size = 224
    epocs = 6
    if torchAvailable:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')