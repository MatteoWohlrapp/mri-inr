'''For now for stuff that has no other place to go. Dont kill me for this'''
import torch

def nan_in_tensor(tensor: torch.Tensor, msg: str) -> None:
    if torch.isnan(tensor).any():
        if msg:
            print(msg)
        print(tensor, flush=True)
        raise ValueError("NaNs in tensor")