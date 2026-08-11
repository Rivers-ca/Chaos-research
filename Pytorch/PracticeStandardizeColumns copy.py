import torch



def standardize(x: torch.Tensor, eps: float = 1e-8):
    if x.dim() != 2:
         raise ValueError(f"expected a 2-d tensorm got shape {tuple(x.shape)}")
    mean = x.mean(dim=0) #creates our mean for the z score formula
    std = x.std(dim=0, unbiased=False) #creates our standard deviation for the z score formula
    safe_stds = std.clamp(min=eps) #Prevents std from being zero to avoid division by zero
    return (x - mean) / safe_stds #z score formula, where ((old value(x) - column mean(mean))/ column standard deviation(std))


if __name__ == "__main__": 
    test = torch.tensor([[1., 10.], [2., 10.], [3., 10.]])
    print(standardize(test))