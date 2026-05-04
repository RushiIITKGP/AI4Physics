import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Generate Data ---
m = 1.0  # Mass
c = 1.0  # Damping Coefficient
k = 20.0 # Spring Constant

def generate_oscillator_data(m=1.0, c=1.0, k=20.0, t_max=10.0, num_points=500):
    omega_0 = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    omega_d = omega_0 * np.sqrt(1 - zeta**2)
    
    t = np.linspace(0, t_max, num_points)
    A = 1.0 
    phi = 0.0
    
    decay = np.exp(-zeta * omega_0 * t)
    x = A * decay * np.cos(omega_d * t + phi)
    v = -A * decay * (zeta * omega_0 * np.cos(omega_d * t + phi) + omega_d * np.sin(omega_d * t + phi))
    a = (-c * v - k * x) / m
    return t, x, v, a

t_data, x_data, v_data, a_data = generate_oscillator_data(m, c, k)

df = pd.DataFrame({
    'time': t_data, 
    'position': x_data, 
    'velocity': v_data, 
    'acceleration': a_data
})
df.to_csv('oscillator_data.csv', index=False)
print("Full dataset successfully saved to 'oscillator_data.csv'")


t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)

# 80% for training, 20% for testing
t_train, t_test, x_train, x_test = train_test_split(
    t_tensor, x_tensor, test_size=0.2, random_state=42
)

t_train.requires_grad_(True)
t_test.requires_grad_(True)

# Shared Brain Setup --------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

normal_mlp = SimpleMLP()
pinn_model = SimpleMLP()


epochs = 1000 # Let's run it 1000 times so it actually learns
optimizer_normal = torch.optim.Adam(normal_mlp.parameters(), lr=0.001)
optimizer_pinn = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
mse_loss_fn = nn.MSELoss()

print("Training models...")
for epoch in range(epochs):
    
    # === Train Normal MLP ===
    optimizer_normal.zero_grad()
    x_pred_normal = normal_mlp(t_train) # Notice we only pass in t_train
    loss_normal = mse_loss_fn(x_pred_normal, x_train)
    loss_normal.backward()
    optimizer_normal.step()
    
    # === Train PINN ===
    optimizer_pinn.zero_grad()
    x_pred_pinn = pinn_model(t_train)
    data_loss_pinn = mse_loss_fn(x_pred_pinn, x_train)
    
    # Extract Velocity & Acceleration
    v_pred = torch.autograd.grad(x_pred_pinn, t_train, grad_outputs=torch.ones_like(x_pred_pinn), create_graph=True)[0]
    a_pred = torch.autograd.grad(v_pred, t_train, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    # Physics Residual
    physics_residual = (m * a_pred) + (c * v_pred) + (k * x_pred_pinn)
    physics_loss = mse_loss_fn(physics_residual, torch.zeros_like(physics_residual))
    
    # Combine (giving physics loss a slightly smaller weight)
    total_loss_pinn = data_loss_pinn + (0.1 * physics_loss)
    total_loss_pinn.backward()
    optimizer_pinn.step()

print("Training complete!\n")

# --- 6. Accuracy Check on Unseen Test Data ---
print("--- TEST DATA EVALUATION ---")
# We test both models on the 20% of data (t_test, x_test) they have never seen before

# Normal MLP Test Check
normal_mlp.eval()
with torch.no_grad():
    test_pred_normal = normal_mlp(t_test)
    test_loss_normal = mse_loss_fn(test_pred_normal, x_test)
print(f"Normal MLP Test Loss (MSE): {test_loss_normal.item():.5f}")

# PINN Test Check
pinn_model.eval()

test_pred_pinn = pinn_model(t_test)
test_data_loss_pinn = mse_loss_fn(test_pred_pinn, x_test)

test_v_pred = torch.autograd.grad(test_pred_pinn, t_test, grad_outputs=torch.ones_like(test_pred_pinn), create_graph=True)[0]
test_a_pred = torch.autograd.grad(test_v_pred, t_test, grad_outputs=torch.ones_like(test_v_pred), create_graph=True)[0]

test_physics_residual = (m * test_a_pred) + (c * test_v_pred) + (k * test_pred_pinn)
test_physics_loss = mse_loss_fn(test_physics_residual, torch.zeros_like(test_physics_residual))

print(f"PINN Data Test Loss (MSE):  {test_data_loss_pinn.item():.5f}")
print(f"PINN Physics Violation:     {test_physics_loss.item():.5f}")



#### Linear Probing ####
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("--- STARTING LINEAR PROBING ---")

# ------ Get the True Velocity Data -----
v_tensor = torch.tensor(v_data, dtype=torch.float32).view(-1, 1)
v_train, v_test = train_test_split(v_tensor, test_size=0.2, random_state=42)


extractor_normal = normal_mlp.net[:4]
extractor_pinn = pinn_model.net[:4]


with torch.no_grad():
    # Record the internal thoughts of both models on the Training data
    h_train_normal = extractor_normal(t_train).numpy()
    h_train_pinn = extractor_pinn(t_train).numpy()
    
    # Record the internal thoughts on the Testing data
    h_test_normal = extractor_normal(t_test).numpy()
    h_test_pinn = extractor_pinn(t_test).numpy()

# ----- Train the Linear Probes -------

probe_normal = LinearRegression()
probe_normal.fit(h_train_normal, v_train.numpy())

probe_pinn = LinearRegression()
probe_pinn.fit(h_train_pinn, v_train.numpy())


v_pred_test_normal = probe_normal.predict(h_test_normal)
v_pred_test_pinn = probe_pinn.predict(h_test_pinn)

# R^2 Score
print("How well did the hidden layers 'know' the True Velocity?")
print(f"Normal MLP Probe R^2 Score: {r2_score(v_test.numpy(), v_pred_test_normal):.4f}")
print(f"PINN Probe R^2 Score:       {r2_score(v_test.numpy(), v_pred_test_pinn):.4f}")

# ------- Extract the Activation Vector! -------

physics_vector = probe_pinn.coef_
print("\nshape of extracted Physics Vector:")
print(physics_vector.shape) 


print("\n--- STARTING ABLATION ---")


vec_normal = torch.tensor(probe_normal.coef_[0], dtype=torch.float32)
vec_pinn = torch.tensor(probe_pinn.coef_[0], dtype=torch.float32)

def ablate_vector(activations, vector):
   
    # h_new = h - ( (h dot v) / (v dot v) ) * v
    v_dot_v = torch.dot(vector, vector)
    
  
    ablated_acts = []
    for h in activations:
        h_dot_v = torch.dot(h, vector)
        projection = (h_dot_v / v_dot_v) * vector
        h_new = h - projection
        ablated_acts.append(h_new)
        
    return torch.stack(ablated_acts)

# Grab the hidden activations (using t_test)
with torch.no_grad():
    h_test_normal = extractor_normal(t_test)
    h_test_pinn = extractor_pinn(t_test)


ablated_h_normal = ablate_vector(h_test_normal, vec_normal)
ablated_h_pinn = ablate_vector(h_test_pinn, vec_pinn)

# Plug the "Muted" brains back into the final layer of the networks
final_layer_normal = normal_mlp.net[4]
final_layer_pinn = pinn_model.net[4]

with torch.no_grad():
    
    ablated_pred_normal = final_layer_normal(ablated_h_normal)
    ablated_pred_pinn = final_layer_pinn(ablated_h_pinn)
    
    # Calculate the new Data Error (MSE)
    ablated_loss_normal = mse_loss_fn(ablated_pred_normal, x_test)
    ablated_loss_pinn = mse_loss_fn(ablated_pred_pinn, x_test)

print("--- RESULTS -----")
print(f"Normal MLP Data Loss AFTER Ablation: {ablated_loss_normal.item():.5f}")
print(f"PINN Data Loss AFTER Ablation:       {ablated_loss_pinn.item():.5f}")






t_test.requires_grad_(True)

# Forward pass up to Layer 3 (keeping the computational graph attached)
h_pinn_grad = extractor_pinn(t_test)

# Differentiable Ablation (The Mute Button, but PyTorch friendly)
vec_pinn_tensor = torch.tensor(probe_pinn.coef_[0], dtype=torch.float32)
v_dot_v = torch.dot(vec_pinn_tensor, vec_pinn_tensor)

# Calculate the projection using matrix math so PyTorch autograd can track it
h_dot_v = torch.matmul(h_pinn_grad, vec_pinn_tensor).unsqueeze(1) 
projection = (h_dot_v / v_dot_v) * vec_pinn_tensor 
ablated_h_pinn_grad = h_pinn_grad - projection

# Final layer prediction using the "Muted" brain
ablated_pred_pinn_grad = final_layer_pinn(ablated_h_pinn_grad)

# Extract the NEW Velocity and Acceleration using Autograd
ablated_v_pred = torch.autograd.grad(
    ablated_pred_pinn_grad, t_test, 
    grad_outputs=torch.ones_like(ablated_pred_pinn_grad), create_graph=True
)[0]

ablated_a_pred = torch.autograd.grad(
    ablated_v_pred, t_test, 
    grad_outputs=torch.ones_like(ablated_v_pred), create_graph=True
)[0]

# Calculate the new Physics Violation
ablated_physics_residual = (m * ablated_a_pred) + (c * ablated_v_pred) + (k * ablated_pred_pinn_grad)
ablated_physics_loss = mse_loss_fn(ablated_physics_residual, torch.zeros_like(ablated_physics_residual))

print(f"PINN Physics Violation BEFORE Ablation:  {test_physics_loss.item():.5f}") 
print(f"PINN Physics Violation AFTER Ablation:   {ablated_physics_loss.item():.5f}")