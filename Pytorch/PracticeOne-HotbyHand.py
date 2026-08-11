import torch

def one_hot(labels: torch.Tensor, num_classes:int) -> torch.Tensor:

    indices = labels.unsqueeze(1) 
    output = torch.zeros(
        (labels.shape[0], num_classes), 
            dtype = torch.float32, 
              device = labels.device) 
    output.scatter_(dim=1, index=indices, value=1.0)
    return (labels.unsqueeze(1) == torch.arange(num_classes, device=labels.device)).float()

if __name__ == "__main__":
    test = torch.tensor([2,0,1], dtype = torch.int64)

    print(one_hot(test, 4))