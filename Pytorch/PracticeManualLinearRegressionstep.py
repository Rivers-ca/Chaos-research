import torch

def sgd_step(X, y, w, b, lr): 
    pred = X @ w + b
    error = pred - y 
    sq_error = error * error 
    loss = sq_error.mean()
    loss.backward()

    with torch.no_grad():
        w.sub_(lr * w.grad)
        b.sub_(lr * b.grad)

    loss_value = loss.item()

    w.grad.zero_()
    b.grad.zero_() 
    
    return loss_value, w, b

if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(100, 3)
    true_w = torch.tensor([2., -1., 0.5])
    y = X @ true_w + 0.7

    w = torch.zeros(3, requires_grad=True)
    b = torch.zeros((), requires_grad=True)

    for i in range(200):
        loss, w, b = sgd_step(X, y, w, b, lr=0.1)
        if i % 50 == 0:
            print(i, loss)
    print(w, b)
