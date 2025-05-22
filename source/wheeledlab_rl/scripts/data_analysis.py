import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the saved data

POLICY = "true-leaf-548"

data_path = "/home/tongo/WheeledLab/source/wheeledlab_rl/logs/"+POLICY+"/playback/play-name-rollouts.pt"
data = torch.load(data_path)

# Convert to numpy for plotting (if needed)
actions = data['actions'].cpu().numpy()            # Shape: [timesteps, num_envs, action_dim]
observations = data['observations'].cpu().numpy()  # Shape: [timesteps, num_envs, obs_dim]
time = data['time'].cpu().numpy()  
s_idx = torch.squeeze(data['s_idx']).cpu().numpy()  

start_idx = 0 
end_idx = 45

plt.figure(figsize=(12, 4))
for env_idx in range(actions.shape[1]):  # Loop through environments
    plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 0], label=f'Env {env_idx} (Throttle)')
    plt.plot(time[start_idx:end_idx], actions[start_idx:end_idx, env_idx, 1], '--', label=f'Env {env_idx} (Steering)')
plt.xlabel("time [s]")
plt.ylabel("Action Value")
plt.title(POLICY+": Policy Actions Over Time")
plt.legend()
plt.grid()
plt.show()

# Example: Plot heading error and lateral deviation
plt.figure(figsize=(12, 4))
plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 0], label='Base Lin Vel X')  # Adjust indices based on your obs space
plt.plot(time[start_idx:end_idx], observations[start_idx:end_idx, 0, 1], label='Base Lin Vel Y')
plt.xlabel("time [s]")
plt.ylabel("velocity [m/s]")
plt.title(POLICY+": Selected Observation Features")
plt.legend()
plt.grid()
plt.show()