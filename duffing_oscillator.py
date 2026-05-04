import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Generate Duffing Oscillator Data ---
m = 1.0  # Mass
c = 0.25  # Damping Coefficient
k = 1.0  # Linear Spring Constant
beta = 5.0  # Nonlinear Spring Constant
F = 0.5  # Driving Force Amplitude
Omega = 1.2  # Driving Frequency

def generate_duffing_data(
    m=1.0,
    c=0.25,
    k=1.0,
    beta=5.0,
    F=0.5,
    Omega=1.2,
    t_max=20.0,
    num_points=800,
    x0=1.0,
    v0=0.0
):
    t = np.linspace(0, t_max, num_points)

    def duffing_rhs(t_current, state):
        x, v = state
        a = (F * np.cos(Omega * t_current) - c * v - k * x - beta * x**3) / m
        return [v, a]

    solution = solve_ivp(
        duffing_rhs,
        t_span=(0.0, t_max),
        y0=[x0, v0],
        t_eval=t,
        rtol=1e-9,
        atol=1e-11
    )

    if not solution.success:
        raise RuntimeError(f"Duffing data generation failed: {solution.message}")

    x = solution.y[0]
    v = solution.y[1]
    a = (F * np.cos(Omega * t) - c * v - k * x - beta * x**3) / m
    return t, x, v, a

t_data, x_data, v_data, a_data = generate_duffing_data(m, c, k, beta, F, Omega)


df = pd.DataFrame({
    'time': t_data, 
    'position': x_data, 
    'velocity': v_data, 
    'acceleration': a_data
})
df.to_csv('duffing_oscillator_data.csv', index=False)
print("Full Duffing oscillator dataset successfully saved to 'duffing_oscillator_data.csv'")

plt.figure(figsize=(10, 5))
plt.plot(t_data, x_data, label='Position x(t)')
plt.plot(t_data, v_data, label='Velocity v(t)', alpha=0.75)
plt.title('Forced Duffing Oscillator Trajectory')
plt.xlabel('Time')
plt.ylabel('State Value')
plt.legend()
plt.tight_layout()
plt.savefig('duffing_trajectory.png', dpi=150)
plt.close()
print("Trajectory plot saved to 'duffing_trajectory.png'")

# Train/Test Split
t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)


t_min, t_max = t_tensor.min(), t_tensor.max()
t_normalized = (t_tensor - t_min) / (t_max - t_min)

# 20% for training (PINNs shine with sparse data), 80% for testing
t_train_norm, t_test_norm, x_train, x_test = train_test_split(
    t_normalized, x_tensor, test_size=0.8, random_state=42
)


t_train_original = t_train_norm * (t_max - t_min) + t_min
t_test_original = t_test_norm * (t_max - t_min) + t_min


t_train_norm.requires_grad_(True)
t_test_norm.requires_grad_(True)
t_train_original.requires_grad_(True)
t_test_original.requires_grad_(True)

# ----- Shared Brain Setup -------
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

# ------ Training Setup --------
epochs = 5000  # Increased from 1000 to 5000 for better convergence
optimizer_normal = torch.optim.Adam(normal_mlp.parameters(), lr=0.001)
optimizer_pinn = torch.optim.Adam(pinn_model.parameters(), lr=0.001)
mse_loss_fn = nn.MSELoss()

# extra physics collocation points
t_physics_norm = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1)
t_physics_norm.requires_grad_(True)
t_physics_original = t_physics_norm * (t_max - t_min) + t_min

def duffing_residual(x_pred, v_pred, a_pred, t_normalized, t_min, t_max):
    
    
    t_original = t_normalized * (t_max - t_min) + t_min
    return (
        (m * a_pred)
        + (c * v_pred)
        + (k * x_pred)
        + (beta * x_pred**3)
        - (F * torch.cos(Omega * t_original))
    )

print("Training models...")
for epoch in range(epochs):

    # === Train Normal MLP ===
    optimizer_normal.zero_grad()
    x_pred_normal = normal_mlp(t_train_norm) # Use normalized time for neural network
    loss_normal = mse_loss_fn(x_pred_normal, x_train)
    loss_normal.backward()
    optimizer_normal.step()

    # === Train PINN ===
    optimizer_pinn.zero_grad()
    x_pred_pinn = pinn_model(t_train_norm)
    data_loss_pinn = mse_loss_fn(x_pred_pinn, x_train)

    # Extract Velocity & Acceleration (using normalized time)
    v_pred = torch.autograd.grad(x_pred_pinn, t_train_norm, grad_outputs=torch.ones_like(x_pred_pinn), create_graph=True)[0]
    a_pred = torch.autograd.grad(v_pred, t_train_norm, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]

    # Duffing Physics Residual on training data
    physics_residual = duffing_residual(x_pred_pinn, v_pred, a_pred, t_train_norm, t_min, t_max)
    physics_loss = mse_loss_fn(physics_residual, torch.zeros_like(physics_residual))

    # Additional physics loss on collocation points
    x_pred_physics = pinn_model(t_physics_norm)
    v_physics = torch.autograd.grad(x_pred_physics, t_physics_norm, grad_outputs=torch.ones_like(x_pred_physics), create_graph=True)[0]
    a_physics = torch.autograd.grad(v_physics, t_physics_norm, grad_outputs=torch.ones_like(v_physics), create_graph=True)[0]
    physics_residual_physics = duffing_residual(x_pred_physics, v_physics, a_physics, t_physics_norm, t_min, t_max)
    physics_loss_physics = mse_loss_fn(physics_residual_physics, torch.zeros_like(physics_residual_physics))


    total_loss_pinn = data_loss_pinn + (1.0 * (physics_loss + physics_loss_physics))
    total_loss_pinn.backward()
    optimizer_pinn.step()


    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Normal Loss={loss_normal.item():.5f}, PINN Data={data_loss_pinn.item():.5f}, PINN Physics={physics_loss.item():.5f}")

print("Training complete!\n")


print("--- TEST DATA EVALUATION ---")
# test both models on the 80% of data (t_test_norm, x_test) 

# Normal MLP Test Check
normal_mlp.eval() 
with torch.no_grad(): 
    test_pred_normal = normal_mlp(t_test_norm)
    test_loss_normal = mse_loss_fn(test_pred_normal, x_test)
print(f"Normal MLP Test Loss (MSE): {test_loss_normal.item():.5f}")

# PINN Test Check
pinn_model.eval()

test_pred_pinn = pinn_model(t_test_norm)
test_data_loss_pinn = mse_loss_fn(test_pred_pinn, x_test)

test_v_pred = torch.autograd.grad(test_pred_pinn, t_test_norm, grad_outputs=torch.ones_like(test_pred_pinn), create_graph=True)[0]
test_a_pred = torch.autograd.grad(test_v_pred, t_test_norm, grad_outputs=torch.ones_like(test_v_pred), create_graph=True)[0]

test_physics_residual = duffing_residual(test_pred_pinn, test_v_pred, test_a_pred, t_test_norm, t_min, t_max)
test_physics_loss = mse_loss_fn(test_physics_residual, torch.zeros_like(test_physics_residual))

print(f"PINN Data Test Loss (MSE):  {test_data_loss_pinn.item():.5f}")
print(f"PINN Physics Violation:     {test_physics_loss.item():.5f}")



#### Linear Probing ####
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("--- STARTING LINEAR PROBING ---")

# --- True Velocity, Acceleration, and Nonlinear Term Data ---

v_tensor = torch.tensor(v_data, dtype=torch.float32).view(-1, 1)
a_tensor = torch.tensor(a_data, dtype=torch.float32).view(-1, 1)
x3_tensor = torch.tensor(x_data**3, dtype=torch.float32).view(-1, 1)  # Nonlinear term

v_train, v_test = train_test_split(v_tensor, test_size=0.8, random_state=42)
a_train, a_test = train_test_split(a_tensor, test_size=0.8, random_state=42)
x3_train, x3_test = train_test_split(x3_tensor, test_size=0.8, random_state=42)

# --- "Brain Scanners" (Extracting Activations) ---

extractor_normal = normal_mlp.net[:4]
extractor_pinn = pinn_model.net[:4]


with torch.no_grad():
  
    h_train_normal = extractor_normal(t_train_norm).numpy()
    h_train_pinn = extractor_pinn(t_train_norm).numpy()

    h_test_normal = extractor_normal(t_test_norm).numpy()
    h_test_pinn = extractor_pinn(t_test_norm).numpy()

# ---Train the Linear Probes ---

# Velocity probes
probe_normal_v = LinearRegression()
probe_normal_v.fit(h_train_normal, v_train.numpy())

probe_pinn_v = LinearRegression()
probe_pinn_v.fit(h_train_pinn, v_train.numpy())

# Acceleration probes
probe_normal_a = LinearRegression()
probe_normal_a.fit(h_train_normal, a_train.numpy())

probe_pinn_a = LinearRegression()
probe_pinn_a.fit(h_train_pinn, a_train.numpy())

# Nonlinear term (x^3) probes
probe_normal_x3 = LinearRegression()
probe_normal_x3.fit(h_train_normal, x3_train.numpy())

probe_pinn_x3 = LinearRegression()
probe_pinn_x3.fit(h_train_pinn, x3_train.numpy())



# Velocity predictions
v_pred_test_normal = probe_normal_v.predict(h_test_normal)
v_pred_test_pinn = probe_pinn_v.predict(h_test_pinn)

# Acceleration predictions
a_pred_test_normal = probe_normal_a.predict(h_test_normal)
a_pred_test_pinn = probe_pinn_a.predict(h_test_pinn)

# Nonlinear term predictions
x3_pred_test_normal = probe_normal_x3.predict(h_test_normal)
x3_pred_test_pinn = probe_pinn_x3.predict(h_test_pinn)

# R^2 Score

print(f"\nVelocity Probing:")
print(f"  Normal MLP Probe R^2 Score: {r2_score(v_test.numpy(), v_pred_test_normal):.4f}")
print(f"  PINN Probe R^2 Score:       {r2_score(v_test.numpy(), v_pred_test_pinn):.4f}")

print(f"\nAcceleration Probing:")
print(f"  Normal MLP Probe R^2 Score: {r2_score(a_test.numpy(), a_pred_test_normal):.4f}")
print(f"  PINN Probe R^2 Score:       {r2_score(a_test.numpy(), a_pred_test_pinn):.4f}")

print(f"\nNonlinear Term (x³) Probing:")
print(f"  Normal MLP Probe R^2 Score: {r2_score(x3_test.numpy(), x3_pred_test_normal):.4f}")
print(f"  PINN Probe R^2 Score:       {r2_score(x3_test.numpy(), x3_pred_test_pinn):.4f}")

# ------ Extract the Activation Vectors! ---

physics_vector_v = probe_pinn_v.coef_
physics_vector_a = probe_pinn_a.coef_
physics_vector_x3 = probe_pinn_x3.coef_

print("\nshapes of extracted Physics Vectors:")
print(f"Velocity Vector:     {physics_vector_v.shape}")  
print(f"Acceleration Vector: {physics_vector_a.shape}")  
print(f"Nonlinear Vector:   {physics_vector_x3.shape}")  


print("\n--- STARTING ABLATION ---")

vec_normal_v = torch.tensor(probe_normal_v.coef_[0], dtype=torch.float32)
vec_pinn_v = torch.tensor(probe_pinn_v.coef_[0], dtype=torch.float32)

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

# Grab the hidden activations (using t_test_norm)
with torch.no_grad():
    h_test_normal = extractor_normal(t_test_norm)
    h_test_pinn = extractor_pinn(t_test_norm)


ablated_h_normal = ablate_vector(h_test_normal, vec_normal_v)
ablated_h_pinn = ablate_vector(h_test_pinn, vec_pinn_v)

# Plug the "Muted" brains back into the final layer of the networks
final_layer_normal = normal_mlp.net[4]
final_layer_pinn = pinn_model.net[4]

with torch.no_grad():
    # What does the network predict now that it's "blind" to velocity?
    ablated_pred_normal = final_layer_normal(ablated_h_normal)
    ablated_pred_pinn = final_layer_pinn(ablated_h_pinn)

    # Calculate the new Data Error (MSE)
    ablated_loss_normal = mse_loss_fn(ablated_pred_normal, x_test)
    ablated_loss_pinn = mse_loss_fn(ablated_pred_pinn, x_test)

print("----- RESULTS ------")
print(f"Normal MLP Data Loss AFTER Ablation: {ablated_loss_normal.item():.5f}")
print(f"PINN Data Loss AFTER Ablation:       {ablated_loss_pinn.item():.5f}")


# ensure t_test_norm is tracking gradients so we can do our physics math
t_test_norm.requires_grad_(True)


h_pinn_grad = extractor_pinn(t_test_norm)

# Differentiable Ablation 
vec_pinn_tensor = torch.tensor(probe_pinn_v.coef_[0], dtype=torch.float32)
v_dot_v = torch.dot(vec_pinn_tensor, vec_pinn_tensor)

h_dot_v = torch.matmul(h_pinn_grad, vec_pinn_tensor).unsqueeze(1)
projection = (h_dot_v / v_dot_v) * vec_pinn_tensor
ablated_h_pinn_grad = h_pinn_grad - projection

#Final layer prediction using the "Muted" brain
ablated_pred_pinn_grad = final_layer_pinn(ablated_h_pinn_grad)

# Extract the NEW Velocity and Acceleration using Autograd
ablated_v_pred = torch.autograd.grad(
    ablated_pred_pinn_grad, t_test_norm,
    grad_outputs=torch.ones_like(ablated_pred_pinn_grad), create_graph=True
)[0]

ablated_a_pred = torch.autograd.grad(
    ablated_v_pred, t_test_norm,
    grad_outputs=torch.ones_like(ablated_v_pred), create_graph=True
)[0]

# Calculate the new Duffing Physics Violation
ablated_physics_residual = duffing_residual(ablated_pred_pinn_grad, ablated_v_pred, ablated_a_pred, t_test_norm, t_min, t_max)
ablated_physics_loss = mse_loss_fn(ablated_physics_residual, torch.zeros_like(ablated_physics_residual))

print(f"PINN Physics Violation BEFORE Ablation:  {test_physics_loss.item():.5f}")
print(f"PINN Physics Violation AFTER Ablation:   {ablated_physics_loss.item():.5f}")


print("\n----- ACCELERATION ABLATION TEST -----")

# Test - removing acceleration also breaks physics compliance
print("Testing hypothesis: Removing acceleration should increase physics loss")


h_pinn_grad_a = extractor_pinn(t_test_norm)

# Differentiable Ablation for Acceleration
vec_pinn_a_tensor = torch.tensor(probe_pinn_a.coef_[0], dtype=torch.float32)
v_dot_v_a = torch.dot(vec_pinn_a_tensor, vec_pinn_a_tensor)

# Calculate the projection for acceleration
h_dot_v_a = torch.matmul(h_pinn_grad_a, vec_pinn_a_tensor).unsqueeze(1)
projection_a = (h_dot_v_a / v_dot_v_a) * vec_pinn_a_tensor
ablated_h_pinn_grad_a = h_pinn_grad_a - projection_a

# Final layer prediction using acceleration-ablated brain
ablated_pred_pinn_grad_a = final_layer_pinn(ablated_h_pinn_grad_a)

# Extract the NEW Velocity and Acceleration using Autograd
ablated_v_pred_a = torch.autograd.grad(
    ablated_pred_pinn_grad_a, t_test_norm,
    grad_outputs=torch.ones_like(ablated_pred_pinn_grad_a), create_graph=True
)[0]

ablated_a_pred_a = torch.autograd.grad(
    ablated_v_pred_a, t_test_norm,
    grad_outputs=torch.ones_like(ablated_v_pred_a), create_graph=True
)[0]

# Calculate the new Duffing Physics Violation
ablated_physics_residual_a = duffing_residual(ablated_pred_pinn_grad_a, ablated_v_pred_a, ablated_a_pred_a, t_test_norm, t_min, t_max)
ablated_physics_loss_a = mse_loss_fn(ablated_physics_residual_a, torch.zeros_like(ablated_physics_residual_a))

print(f"PINN Physics Violation BEFORE Acceleration Ablation: {test_physics_loss.item():.5f}")
print(f"PINN Physics Violation AFTER Acceleration Ablation:  {ablated_physics_loss_a.item():.5f}")

# Calculate the increase factor
if test_physics_loss.item() > 0:
    increase_factor_a = ablated_physics_loss_a.item() / test_physics_loss.item()
    print(f"Physics Violation Increase Factor: {increase_factor_a:.1f}x")
else:
    print("Physics violation was already near zero, cannot calculate increase factor")

# Compare with velocity ablation
print(f"\nComparison:")
print(f"Velocity Ablation Physics Increase:    {ablated_physics_loss.item() / test_physics_loss.item():.1f}x")
print(f"Acceleration Ablation Physics Increase: {increase_factor_a:.1f}x")

if increase_factor_a > 10:
    print("\nCONFIRMED: Acceleration ablation dramatically increases physics loss!")
    print("  This proves acceleration is encoded in the PINN's hidden layers.")
else:
    print("\nNote: Acceleration ablation shows smaller increase than velocity ablation")
    print("  This suggests acceleration encoding may be more distributed or different in nature")

# --- EXTRAPOLATION TEST ---
print("\n--- EXTRAPOLATION TEST (BEYOND TRAINING RANGE) ---")

t_extrap_norm = torch.linspace(1.0, 1.5, 100, dtype=torch.float32).view(-1, 1)
t_extrap_norm.requires_grad_(True)

# Generate true Duffing data for extrapolation range
t_extrap_original = t_extrap_norm * (t_max - t_min) + t_min
t_extrap_np = t_extrap_original.detach().numpy().flatten()

# Solve the ODE for extrapolation range
def duffing_rhs_np(t_current, state):
    x, v = state
    a = (F * np.cos(Omega * t_current) - c * v - k * x - beta * x**3) / m
    return [v, a]

# Get the final state from training data as initial condition
x_final = x_data[-1]
v_final = v_data[-1]

solution_extrap = solve_ivp(
    duffing_rhs_np,
    t_span=(t_data[-1], t_extrap_np[-1]),
    y0=[x_final, v_final],
    t_eval=t_extrap_np,
    rtol=1e-9,
    atol=1e-11
)

x_extrap_true = torch.tensor(solution_extrap.y[0], dtype=torch.float32).view(-1, 1)

# Test extrapolation
normal_mlp.eval()
pinn_model.eval()

with torch.no_grad():
    x_pred_normal_extrap = normal_mlp(t_extrap_norm)
    x_pred_pinn_extrap = pinn_model(t_extrap_norm)

# Calculate extrapolation errors
extrap_loss_normal = mse_loss_fn(x_pred_normal_extrap, x_extrap_true)
extrap_loss_pinn = mse_loss_fn(x_pred_pinn_extrap, x_extrap_true)

print(f"Normal MLP Extrapolation Loss: {extrap_loss_normal.item():.5f}")
print(f"PINN Extrapolation Loss:       {extrap_loss_pinn.item():.5f}")

# Check physics violation on extrapolation
x_pred_pinn_extrap_grad = pinn_model(t_extrap_norm)
v_pred_extrap = torch.autograd.grad(x_pred_pinn_extrap_grad, t_extrap_norm, grad_outputs=torch.ones_like(x_pred_pinn_extrap_grad), create_graph=True)[0]
a_pred_extrap = torch.autograd.grad(v_pred_extrap, t_extrap_norm, grad_outputs=torch.ones_like(v_pred_extrap), create_graph=True)[0]

physics_residual_extrap = duffing_residual(x_pred_pinn_extrap_grad, v_pred_extrap, a_pred_extrap, t_extrap_norm, t_min, t_max)
physics_loss_extrap = mse_loss_fn(physics_residual_extrap, torch.zeros_like(physics_residual_extrap))

print(f"PINN Physics Violation on Extrapolation: {physics_loss_extrap.item():.5f}")


