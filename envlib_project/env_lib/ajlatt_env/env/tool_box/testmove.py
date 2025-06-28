import torch
import matplotlib.pyplot as plt

# Parameters
k_p = 1.0  # Proportional gain for y position control
k_theta = 0.5  # Proportional gain for heading control
dt = 0.1  # Time step
t_end = 20  # Simulation time
v_nominal = 0.5  # Nominal velocity for forward motion

# Initial state [x^1, x^2, theta] as a PyTorch tensor
x = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

# Utility function to wrap angle to [-pi, pi] using PyTorch
def pi_to_pi(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

# Define the trajectory function (e.g., sine wave)
def func(x1):
    return torch.sin(x1)  # Change this to any trajectory you want

# Control law to make the system follow the given trajectory x^2 = func(x^1)
def control_law(x, func):
    # Desired trajectory: x^2 = func(x^1)
    x1_desired = x[0]
    x2_desired = func(x1_desired)

    # Error in x^2 direction
    e = x[1] - x2_desired

    # Control law for forward velocity v
    v = v_nominal  # Nominal forward velocity
    
    # Desired heading to follow the trajectory
    theta_desired = torch.atan(torch.cos(x1_desired))
    
    # Control law for angular velocity omega
    omega = k_theta * (theta_desired - x[2])
    
    return torch.tensor([v, omega])

# Render function to show the trajectory in real-time
def render(positions, desired_trajectory):
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x^1 Position')
    ax.set_ylabel('x^2 Position')
    ax.set_title('System Trajectory Following x^2 = func(x^1)')
    
    system_trajectory, = ax.plot([], [], label="System Trajectory", color='blue')
    desired_traj, = ax.plot(desired_trajectory[:, 0].cpu().numpy(), desired_trajectory[:, 1].cpu().numpy(), label="Desired Trajectory", linestyle='--', color='green')
    ax.legend()
    plt.grid(True)

    for pos in positions:
        system_trajectory.set_data(positions[:len(system_trajectory.get_xdata()) + 1, 0].cpu().numpy(), positions[:len(system_trajectory.get_xdata()) + 1, 1].cpu().numpy())
        plt.draw()
        plt.pause(0.1)  # Small delay for animation effect

    plt.ioff()  # Disable interactive mode
    plt.show()

# Simulation
positions = []  # Store positions for plotting
time_values = torch.arange(0, t_end, dt)

for t in time_values:
    u = control_law(x, func)  # Get control inputs (v, omega)
    
    # Update state using the system dynamics
    diff = torch.tensor([
        u[0] * dt * torch.cos(x[2]),  # x^1 update
        u[0] * dt * torch.sin(x[2]),  # x^2 update
        u[1] * dt                     # theta update
    ])
    
    x = x + diff
    x[2] = pi_to_pi(x[2])  # Keep theta in [-pi, pi]
    
    positions.append(x[:2])  # Store the (x^1, x^2) position

# Convert positions to a tensor for easier plotting
positions = torch.stack(positions)

# Desired trajectory for comparison
x1_values = positions[:, 0]
x2_desired = func(x1_values)  # Use the function to get the desired x^2 values
desired_trajectory = torch.stack((x1_values, x2_desired), dim=1)

# Render the plot
render(positions, desired_trajectory)
