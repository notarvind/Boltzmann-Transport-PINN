import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
  def __init__(self):
    super(PINN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(2, 16), #input layer, 2 neurons for r, mu
      nn.Tanh(),
      nn.Linear(16, 16), #hidden layer
      nn.Tanh(),
      nn.Linear(16, 1) #output layer, 1 neuron
    )

  def forward(self, x):
    return self.net(x)
#sets up network

num_points = 10000
r = torch.rand(num_points, 1) * 9 + 1 #let r be b/w 1 and 10
mu = torch.rand(num_points, 1) * 2 - 1 #mu = cos(theta), so b/w -1 and 1
inputs = torch.cat([r, mu], dim=1).requires_grad_() #combines both into an array of the form [r, mu], tells PyTorch to track gradients wrt inputs
#collocation points to regulate the network at random points as a check

def j(r): #emissivity
  return torch.exp(-r/5)

def k(r): #opacity
  return 0.3 + (0.2*torch.exp(-r/2))
#realistic eqns for change in emissivity and opacity with radius (exponential decay), but normalised to domain

num_bc = 1000

mu_bc = torch.linspace(-1, 1, num_bc).unsqueeze(1) #shape: [N_bc, 1]
r_bc = torch.full_like(mu_bc, 10) #same shape, BC at 10 i.e, outer surface of star where we assume no neutrino diffusion from
x_bc = torch.cat([r_bc, mu_bc], dim=1) #combines both into an array to get [r, mu] input points of shape [N_bc, 2] - first column = 100 radii, 2nd column = 100 mu
model = PINN() #sets up network
f_pred_bc = model(x_bc) #runs BC array thru model
f_target_bc = torch.zeros_like(f_pred_bc) #identical tensor to f_pred_bc, filled with 0s (which is the target/correct value at the BCs)
loss_MSE = nn.MSELoss() #sets up MSE loss function
bc_loss = loss_MSE(f_pred_bc, f_target_bc) #mean squared error loss due to BCs (to be used later in total loss)
#loss due to boundary conditions

w1 = 1
w2 = 0.5
#weights

def compute_residual_loss(f_pred, inputs):
  r_residual = torch.rand(num_points, 1) * 9 + 1 #let r be b/w 1 and 10, same as before
  mu_residual = torch.rand(num_points, 1) * 2 - 1 #mu = cos(theta), so b/w -1 and 1, same as before
  X_residual = torch.cat([r_residual, mu_residual], dim=1).requires_grad_(True) #combines both into an array of the form [r, mu], tells PyTorch to track gradients wrt inputs
  f_pred = model(X_residual) #same, gets f values for residuals (should be 0)

#lifted autograd bit from gpt cos i had no idea

  df_dr = torch.autograd.grad(
      f_pred, X_residual, grad_outputs=torch.ones_like(f_pred),
      create_graph=True, retain_graph=True
  )[0][:, 0].unsqueeze(1)  # derivative w.r.t. r (first column)
  #calculates partial df/dr using autograd

  df_dmu = torch.autograd.grad(
      f_pred, X_residual, grad_outputs=torch.ones_like(f_pred),
      create_graph=True, retain_graph=True
  )[0][:, 1].unsqueeze(1)  # derivative w.r.t. mu (second column)
  #calculates partial df/dmu using autograd

  residual = mu_residual * df_dr + (1 - mu_residual**2)/r_residual * df_dmu + k(r_residual) * f_pred - j(r_residual) #residual, difference b/w LHS and RHS of eqn
  residual_loss = torch.mean(residual**2) #squared residuals

  return residual_loss

  total_loss = w1*bc_loss + w2*residual_loss #weighted

#note, all of these are part of the untrained model, losses need to be calculated again in training loop to update weights with every epoch

model = PINN()
optimiser = optim.Adam(model.parameters(), lr=0.001) #Adam optimiser, learning rate = 0.001

epoch_num = 5000
for epoch in range(epoch_num):
  optimiser.zero_grad() #clears old grads

  f_pred_bc = model(x_bc)  #Predicts at BCs again using current weights
  bc_loss = loss_MSE(f_pred_bc, f_target_bc)

  residual_loss = compute_residual_loss(model, num_points)

  total_loss = w1 * bc_loss + w2 * residual_loss

  total_loss.backward() #back propagation, calculates how much weights should change

  optimiser.step() #updates weights using gradients computed in back propagation

  if epoch % 100 == 0:
        print(f"Epoch {epoch} | Total: {total_loss.item():.5f} | BC: {bc_loss.item():.5f} | Residual: {residual_loss.item():.5f}")

import numpy as np
import matplotlib.pyplot as plt

r_values = torch.linspace(1, 10, 100)
mu_values = torch.linspace(-1, 1, 100)

R, MU = torch.meshgrid(r_values, mu_values, indexing='ij') #100x100 tensor - every possible value of r with every possible value of mu
grid = torch.cat([R.reshape(-1, 1), MU.reshape(-1, 1)], dim=1) #reshapes into a 10000x2 tensor (100*100) x 2
with torch.no_grad():
    f_vals = model(grid) #to predict without calculating grads

f_vals = f_vals.reshape(100, 100) #reshapes back, for plotting

plt.figure(figsize=(8, 6))
plt.contourf(R.numpy(), MU.numpy(), f_vals.numpy(), levels=50, cmap='viridis')
plt.xlabel('r')
plt.ylabel('μ')
plt.title('Predicted f(r, μ)')
plt.colorbar()
plt.show()

#THE FINAL PLOT IS WRONG, STILL A WORK IN PROGRESS
