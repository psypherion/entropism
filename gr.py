import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# Physical Constants and Lensing Parameters
# -------------------------------------
G = 6.67430e-11          # gravitational constant (m^3 kg^-1 s^-2)
c = 3e8                  # speed of light (m/s)
M = 1e30                 # mass of the lens (kg)
D_L = 1e20               # distance to lens (m)
D_S = 2e20               # distance to source (m)
D_LS = D_S - D_L         # distance between lens and source (m)

# Einstein radius in radians for a point-mass lens:
theta_E = np.sqrt((4 * G * M / c**2) * (D_LS / (D_L * D_S)))
print(f"Einstein radius (theta_E): {theta_E:.3e} rad")

# -------------------------------------
# Simulation Parameters for Source Rays
# -------------------------------------
num_rays = 100000         # number of simulated light rays (each ray = a small process)
beta_range = 5 * theta_E  # angular extent of the source in radians
beta = np.random.uniform(-beta_range, beta_range, num_rays)  # source positions (rays)

# -------------------------------------
# Fundamental Amplitude and Observational Coefficient
# -------------------------------------
# Assume uniform fundamental amplitude: c(beta) = 1 for all rays.
c_beta = np.ones_like(beta)
# Define the observational coefficient: O(beta) = 1 + a*(|beta|/beta_range)
a = 0.3  # modulation factor
O_beta = 1 + a * (np.abs(beta) / beta_range)
# The effective weight for each ray:
weights = O_beta * c_beta  # effectively, just O_beta since c_beta = 1

# -------------------------------------
# Standard Gravitational Lensing Simulation
# -------------------------------------
# The lens equation (in angular units) for a point mass lens:
#   θ - β = θ_E^2 / θ  =>  θ^2 - βθ - θ_E^2 = 0
# Solve the quadratic for each ray:
def lens_equation(beta, theta_E):
    discriminant = np.sqrt(beta**2 + 4 * theta_E**2)
    theta_plus = (beta + discriminant) / 2
    theta_minus = (beta - discriminant) / 2
    return theta_plus, theta_minus

# Compute the two image positions for each ray:
theta_plus, theta_minus = lens_equation(beta, theta_E)
# Combine both images:
theta_obs_standard = np.concatenate([theta_plus, theta_minus])

# Build a histogram to approximate the standard intensity distribution:
num_bins = 1000
bins = np.linspace(-beta_range, beta_range, num_bins)
I_standard, _ = np.histogram(theta_obs_standard, bins=bins, density=True)

# -------------------------------------
# Modified Simulation: Incorporating the Observational Coefficient
# -------------------------------------
# In our framework, each ray's contribution is weighted by its effective weight.
# Since each ray produces two images, the same weight applies to both.
theta_obs_modified = np.concatenate([theta_plus, theta_minus])
weights_total = np.concatenate([weights, weights])
I_modified, _ = np.histogram(theta_obs_modified, bins=bins, weights=weights_total, density=True)

# Normalize the intensity distributions for comparison:
I_standard_norm = I_standard / np.max(I_standard)
I_modified_norm = I_modified / np.max(I_modified)

# -------------------------------------
# Compute Shannon-like Entropy for the Modified Intensity Distribution
# -------------------------------------
# Treat the normalized intensity as a probability distribution.
p = I_modified_norm / np.sum(I_modified_norm)
epsilon = 1e-12  # small constant to avoid log(0)
entropy_modified = -np.sum(p * np.log(p + epsilon))
print(f"Estimated Shannon Entropy (Modified): {entropy_modified:.6f}")

# -------------------------------------
# Plotting the Results
# -------------------------------------
bin_centers = 0.5 * (bins[:-1] + bins[1:])

plt.figure(figsize=(10, 6))
plt.plot(bin_centers*1e3, I_standard_norm, label='Standard GR Lensing', color='blue')
plt.plot(bin_centers*1e3, I_modified_norm, label='Modified (with O(β))', color='red', linestyle='--')
plt.xlabel('Angular Position (milli-radians, scaled)')
plt.ylabel('Normalized Intensity')
plt.title('Gravitational Lensing: Standard vs. Modified Intensity Distribution')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------
# Numerical Analysis: Compare Standard and Modified Models
# -------------------------------------
mse = np.mean((I_standard_norm - I_modified_norm)**2)
rmse = np.sqrt(mse)
corr_coef = np.corrcoef(I_standard_norm, I_modified_norm)[0, 1]

print("Numerical Analysis of Lensing Intensity Patterns:")
print("--------------------------------------------------")
print(f"Mean Squared Error (MSE): {mse:.6e}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6e}")
print(f"Correlation Coefficient: {corr_coef:.6f}")


"""
output : 

Einstein radius (theta_E): 3.851e-09 rad
Estimated Shannon Entropy (Modified): 6.378290
Numerical Analysis of Lensing Intensity Patterns:
--------------------------------------------------
Mean Squared Error (MSE): 1.471035e-04
Root Mean Squared Error (RMSE): 1.212862e-02
Correlation Coefficient: 0.997695
"""
