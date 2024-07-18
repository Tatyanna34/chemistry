# chemistryimport numpy as np
from scipy.linalg import eigh

# Constants
alpha = 1.0   # Parameter for potential (in atomic units)

# Atomic orbital basis functions
def atomic_orbital(r, n, l):
    # Radial part of the wavefunction for hydrogen-like atom
    R_nl = np.sqrt((2.0/n)**3 * np.math.factorial(n-l-1)/(2.0 * n * np.math.factorial(n+l)**3)) * np.exp(-r/n) * (2.0 * r / n)**l
    return R_nl

# Potential energy function (in atomic units)
def potential(r):
    return -alpha / r

# Solve for a single hydrogen-like atom
def solve_hydrogen_atom():
    # Discretization parameters
    r_min = 1e-6  # Avoid singularity at r=0
    r_max = 50.0
    num_points = 1000
    r = np.linspace(r_min, r_max, num_points)
    dr = r[1] - r[0]

    # Construct the Hamiltonian matrix
    H = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                H[i, j] = -0.5 / dr**2 + potential(r[i])
            else:
                H[i, j] = 1.0 / abs(r[i] - r[j])

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(H)

    # Ground state energy and wavefunction
    ground_state_energy = eigenvalues[0]
    ground_state_wavefunction = eigenvectors[:, 0]

    # Normalize the wavefunction
    normalization = np.sqrt(np.trapz(ground_state_wavefunction**2 * r**2, r))
    ground_state_wavefunction /= normalization

    return ground_state_energy, ground_state_wavefunction, r

# Solve for the hydrogen molecule (H2)
def solve_hydrogen_molecule():
    # Solve for two hydrogen atoms
    energy1, wavefunction1, r = solve_hydrogen_atom()
    energy2, wavefunction2, r = solve_hydrogen_atom()

    # Combine the wavefunctions (molecular orbitals)
    molecular_wavefunction = (wavefunction1 + wavefunction2) / np.sqrt(2.0)

    # Calculate the total energy (approximate)
    total_energy = energy1 + energy2

    return total_energy, molecular_wavefunction, r

# Main program
if __name__ == "__main__":
    # Solve for the hydrogen molecule (H2)
    total_energy, molecular_wavefunction, r = solve_hydrogen_molecule()

    # Output results
    print(f"Total energy of the hydrogen molecule: {total_energy:.6f} atomic units")

    # Plot the molecular wavefunction (optional)
    import matplotlib.pyplot as plt
    plt.plot(r, molecular_wavefunction, label='Molecular Wavefunction')
    plt.xlabel('Distance (atomic units)')
    plt.ylabel('Wavefunction Amplitude')
    plt.title('Hydrogen Molecule Wavefunction')
    plt.legend()
    plt.show()
