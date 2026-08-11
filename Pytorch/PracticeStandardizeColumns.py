import torch 

def standardize(x: torch.Tensor, eps: float = 1e-8):
    if x.dim() != 2:
        raise ValueError(f"Value Error expected a 2d shape but got {tuple(x.shape)}")
    mean = x.mean(dim = 0)
    std = x.std(dim = 0, unbiased = False)
    safe_std = std.clamp(min = eps)
    return (x - mean) / safe_std

if __name__ == "__main__":

    test = torch.tensor([[1., 10.,], 
                         [2., 10.], 
                         [3., 10.]])

    print(standardize(test))