# TODO: create undistributed, unsketched version which converges (ugh)

import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.randn(1000, 200)
y = torch.zeros(1000, 1)

model = torch.nn.Sequential(torch.nn.Linear(200, 1))

opt = torch.optim.SGD(model.parameters(), momentum=0, lr=0.0001)

loss_fn = nn.MSELoss()

for i in range(100):
    opt.zero_grad()

    y_pred = model(X)
    loss = loss_fn(y, y_pred)

    print("[{}] Loss: {}".format(i, loss.item()))

    loss.backward()

    opt.step()