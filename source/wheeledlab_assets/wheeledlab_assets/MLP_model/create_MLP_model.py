import torch
import torch.nn as nn

# Define a tiny MLP (input_size=4 for [pos_err(t), pos_err(t-1), vel(t), vel(t-1)])
class DummyThrottleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4),  # Minimal viable model
            nn.Tanh(),        # Outputs in [-1, 1]
            nn.Linear(4, 1),  # Output: 1 torque per joint (CHANGED THIS LINE)
        )
    
    def forward(self, x):
        return self.net(x) * 0.25  # Small torque outputs (0.25 Nm)


class DummyResidualDynamicsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 6),  # Minimal viable model, input are v_x, v_y, yaw_dot, acc_c, steer_c
            nn.Tanh(),        # Outputs in [-1, 1]
            nn.Linear(6, 3),  # Output: 1 torque per joint (CHANGED THIS LINE)
        )
    
    def forward(self, x):
        return self.net(x) * 0.25  # Small torque outputs (0.25 Nm)

# Export as TorchScript
model = DummyResidualDynamicsMLP()
scripted_model = torch.jit.script(model)
scripted_model.save("dummy_residual_dynamics_mlp.pt")  # Save for Isaac Lab