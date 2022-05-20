import torch

class paramStore:
    class default:
        batch_size = 16
        img_size = 224
        epocs = 6
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class cpu:
        batch_size = 16
        img_size = 224
        epocs = 6
        device = torch.device('cpu')