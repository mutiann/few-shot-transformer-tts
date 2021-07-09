import torch

def dict_send_to(data, device, detach=False):
    result = {}
    for key in data:
        t = data[key]
        if isinstance(t, torch.Tensor):
            if detach:
                t = t.detach()
            t = t.to(device)
        result[key] = t
    return result