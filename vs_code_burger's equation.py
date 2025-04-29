# %%
import torch
import torch.nn as nn
import numpy as np
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
            
        )
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim = 1)
        u = self.hidden(inputs)
        return u
    print("Hiefere")

# %%
# define the pde residual function
def pde_residual(x, t, model, nu = 0.01):
    
    x.requires_grad = True
    t.requires_grad = True
    
    u = model(x, t) #predict u from the model
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]  
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]  
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]  

    
    residual = u_t + u * u_x - nu * u_xx 
    return residual

#initial and boundary condiitons
def initial_condition(x):
    return -torch.sin(np.pi * x)

def boundary_condition(x, t):
    return torch.zeros_like(t)
print("2")
    
    

# %%

x = torch.linspace(-1, 1, 200).view(-1, 1)  
t = torch.linspace(0, 1, 100).view(-1, 1) 
x_train, t_train = torch.meshgrid(x.squeeze(), t.squeeze(), indexing = 'xy')
x_train = x_train.reshape(-1, 1)
t_train = t_train.reshape(-1, 1)
print("hi")

# %%
#define the model, optimizer, and loss function
import torch.optim as optim
import torch.nn as nn
loss_fn = nn.MSELoss()


model = PINN()
optimizer = optim.Adam(model.parameters(), lr = 0.01)


# %%
num_epochs = 12000
print("Hi")
for each in range(num_epochs): 
    model.train()
    
    # Compute initial condition loss
    u_pred = model(x, torch.zeros_like(x)) 
    u_true = initial_condition(x)
    loss_ic = torch.mean((u_pred - u_true)**2)
    
    # Compute boundary condition loss
    u_pred_left = model(torch.full_like(t, -1), t) # u(-1, t)
    u_pred_right = model(torch.full_like(t, 1), t) # u(1, t)
    loss_bc = torch.mean((u_pred_left - boundary_condition(torch.full_like(t, -1), t))**2) + \
              torch.mean((u_pred_right - boundary_condition(torch.full_like(t, 1), t))**2)
    
    # Compute PDE residual loss
    residual = pde_residual(x_train, t_train, model)
    loss_pde = torch.mean(residual**2) 
    
    loss = loss_ic + loss_bc + loss_pde
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#     print("Hi")
   
    if each % 500 == 0:
        print(f'Epoch [{each + 1}/{num_epochs}], Loss: {loss.item():.4f}')

      


# %%
# evaluate the model on a grid of (x, t) values
x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.linspace(0, 1, 100).view(-1, 1)
x_test, t_test  = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing = 'xy')
x_test = x_test.reshape(-1, 1)
t_test = t_test.reshape(-1, 1)

model.eval()
with torch.no_grid():
    u_pred = model(x_text, t_text).nump()
    
# reshape the predicted u values for contour plotting
x_test = x_test.numpy().reshape(100, 100)
t_test = t_test.numpy().reshape(100, 100)
u_pred = u_pred.reshape(100, 100)

#plot the PINN solution as a contour plot
plt.figure(figsize = (12, 4))
plt.contourf(x_test, t_test, u_pred, levels = 250, cmap = 'jet')
plt.colorbar()
plt.xlabel('x')
pl.ylabel('t')
plt.title('solution of burgers\' equation using PINNs')
plt.show()


# %%


# %%


# %%


# %%


# %%



