import torch 


def grad_of(x_val: float) -> float: 
    tens = torch.tensor(x_val, dtype = torch.float32, requires_grad = True)
    y = (tens ** 3) + (2 * tens) 
    y.backward()
    return tens.grad.item()


if __name__ == "__main__":
    print(grad_of(3.0))
    