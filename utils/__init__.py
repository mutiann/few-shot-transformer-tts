import torch

def dict_send_to(data, device, detach=False, as_numpy=False):
    result = {}
    for key in data:
        t = data[key]
        if isinstance(t, torch.Tensor):
            if detach:
                t = t.detach()
            t = t.to(device)
            if as_numpy:
                t = t.numpy()
        result[key] = t
    return result