from pyexpat import model

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# # x = torch.tensor(3.0, requires_grad=True)

# # lr = 0.5


# # #Gradient descent loop
# # x = torch.tensor(3.0, requires_grad=True)

# # for i in range(10):
# #     y = x**2 #loss
# #     y.backward() #Gradient into x.grad
# #     with torch.no_grad():
# #         assert x.grad is not None
# #         x -= lr * x.grad
# #     assert x.grad is not None    
# #     x.grad.zero_()
# #     print(f"step {i}: x - {x.item():.4f}")
# # #Stochastic Gradent Descent loop
# # x = torch.tensor(3.0, requires_grad=True)
# # optimizer = torch.optim.SGD([x], lr=lr) # Intialize the optimizer with the parameters to optimize and the learning rate

# # for i in range(10):
# #     y =x**2 #Loss
# #     y.backward()#Gradient into x.grad
# #     optimizer.step() # This is what x-= ly *.grad(side note this is the same as x=x-lr*x.grad)
# #     optimizer.zero_grad() # This is what x.grad.zero_() does
# #     print(f"step {i}: x - {x.item():.4f}")






# # GD vs SGD 
# # fake data: points that roughly follow y = 2x + 1

# xs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])      # note: column shape now
# ys = torch.tensor([[3.1], [4.9], [7.2], [8.8]])
#   # ≈ 2x+1, with a little noise

# model = torch.nn.Linear(1, 1)  # one in and one out
# w, b = model.parameters()       # model's weights and bias
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # learning rate
# loss_fn = nn.MSELoss()                                # this IS ((pred-y)**2).mean()

# for step in range(500):
#     y_pred = model(xs)                     # model's guess for all points
#     loss = loss_fn(y_pred, ys)      # mean squared error
#     loss.backward()                        # gradients into w.grad, b.grad
#     optimizer.step()                       # nudge w and b downhill
#     optimizer.zero_grad()                  # empty the buckets
#     if step % 100 == 0:
#         print(f"step {step}: w={w.item():.3f}, b={b.item():.3f}, loss={loss.item():.3f}")
# w, b = model.weight.item(), model.bias.item()
# print(f"learned: w={w:.3f}, b={b:.3f}")



xs = torch.linspace(-3, 3, 100).reshape(-1, 1)  
ys = xs**2  # y = x^2


model = nn.Sequential(
    nn.Linear(1, 64),  
    nn.ReLU(),        
    nn.Linear(64, 1)   
)

loss_fn = nn.MSELoss()  
optimzer = torch.optim.Adam(model.parameters(), lr=0.01)  

for step in range(1000):
    y_pred = model(xs)
    loss = loss_fn(y_pred, ys) 
    loss.backward() 
    optimzer.step()  
    optimzer.zero_grad()  

    if step % 400 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

with torch.no_grad():
    preds = model(xs)
plt.scatter(xs, ys, s=10, label="true y=x²")
plt.plot(xs, preds, color="red", label="network's fit")
plt.legend(); plt.show()