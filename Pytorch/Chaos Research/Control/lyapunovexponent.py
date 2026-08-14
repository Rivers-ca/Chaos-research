import torch, math 

r = 4.0 

def grad_after_N(N, x0_val=0.4):
    x = torch.tensor(x0_val, requires_grad=True)
    y = x
    for _ in range(N):
        y = r * y * (1 - y)   # logistic map step
    y.backward()
    return abs(x.grad.item()) if x.grad is not None else 0.0

for N in [1, 5, 10, 20, 30, 40]:
    print(f"{N:>3} steps:  |grad| = {grad_after_N(N):.3e}")