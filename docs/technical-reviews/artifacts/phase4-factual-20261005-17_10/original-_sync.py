def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()
