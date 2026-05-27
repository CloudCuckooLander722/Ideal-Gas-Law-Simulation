import matplotlib.pyplot as plt
import kinetics
from kinetics import compute_pressure, initialize_system, update_positions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from vpython import box, canvas, sphere, vector, color, rate
from time import *

def plot_pressure(N, kB, m, L, V, temperatures):
    # --- Simulation Parameters ---
    N = 1000          # Number of particles
    kB = 1.3806e-23   # True Boltzmann constant (J/K)
    m = 6.64e-26      # Mass of Helium particle (kg)
    L = 10.0          # Box width (meters)
    V = L**3          # Volume of the cube container (m^3)
    
    temperatures = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    pressures = []

   
    print("Collecting pressure data across temperatures...")
    
    for t in temperatures:
        pressure = compute_pressure(N=N, kB=kB, m=m, L=L, T=t)
        pressures.append(pressure)

    temperatures = np.array(temperatures)
    pressures = np.array(pressures)

    slope, intercept = np.polyfit(temperatures, pressures, 1)

    sim_kB = (slope * V) / N

    r_constant = sim_kB * 6.022e23

    print("\n=== FINAL ANALYSIS ===")
    print(f"Linear Fit Equation: P = {slope:.4e} * T + {intercept:.4e}")
    print(f"Theoretical Slope (N * kB / V): {(N * kB) / V:.4e}")
    print(f"Simulation Gas Constant (kB):   {r_constant:.4e}")
    
    plt.figure(figsize=(8, 5))
    
    
    plt.scatter(temperatures, pressures, color='crimson', edgecolor='black', s=50, label='Simulation Data', zorder=3)
    
    
    fit_line = slope * temperatures + intercept
    plt.plot(temperatures, fit_line, color='navy', linestyle='--', linewidth=2, label=f'Linear Fit (Slope: {slope:.4e})')
    
    
    plt.title('Verification of Ideal Gas Law: Pressure vs. Temperature', fontsize=12, fontweight='bold')
    plt.xlabel('Temperature T (Kelvin)', fontsize=10)
    plt.ylabel('Pressure P (Pascal)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    
    
    plt.savefig('ideal_gas_law_proof.png', dpi=300)
    print("\nGraph saved successfully as 'ideal_gas_law_proof.png'!")
    plt.show()



def animate_gas(N, kB, m, L, V, temperatures):
    if temperatures is None:
        temperatures = [100, 200, 300, 400, 500]

    scene = canvas(title="Ideal Gas Law Simulation", width=800, height=600)
    
    half_L = L / 2
    floor = box(pos=vector(0, -half_L, 0), length=L, width=L, height=0.1, color=color.white)
    ceiling = box(pos=vector(0, half_L, 0), length=L, width=L, height=0.1, color=color.white)
    backWall = box(pos=vector(0, 0, -half_L), length=L, width=0.1, height=L, color=color.white)
    leftWall = box(pos=vector(-half_L, 0, 0), length=0.1, width=L, height=L, color=color.white)
    rightWall = box(pos=vector(half_L, 0, 0), length=0.1, width=L, height=L, color=color.white)
    front_wall = box(pos=vector(0, 0, half_L), length=L, width=0.1, height=L, color=color.white, opacity=0.0)

    spheres = [sphere(radius=0.2, color=color.cyan, pos=vector(0, 0, 0)) for _ in range(N)]

    steps_per_temp = 300

    for t in temperatures:
        print(f"Simulating temperature: {t} K")

        positional_vector, velocity_vector = initialize_system(N=N, L=L, kB=kB, T=t, m=m)

        safety_factor = 0.01 
        max_speed = np.max(np.linalg.norm(velocity_vector, axis=1))
        dt = (safety_factor * L) / max_speed

        for step in range(steps_per_temp):
            rate(60)
            positional_vector += (velocity_vector * dt)

            for axis in range(3):
                hit_lower = positional_vector[:, axis] <= -half_L
                hit_upper = positional_vector[:, axis] >= half_L

                if np.any(hit_upper):
                    velocity_vector[hit_upper, axis] *= -1
                    positional_vector[hit_upper, axis] = half_L - 0.001

                elif np.any(hit_lower):
                    velocity_vector[hit_lower, axis] *= -1
                    positional_vector[hit_lower, axis] = -half_L - 0.001

            for idx, s in enumerate(spheres):
                s.pos = vector(
                    positional_vector[idx, 0],
                    positional_vector[idx, 1],
                    positional_vector[idx, 2]
                )


                
                    
                


            
            

if __name__ == "__main__":
    N = 1000          
    kB = 1.3806e-23   
    m = 6.64e-26      
    L = 10.0          
    V = L**3          
    
    temperatures = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    animate_gas(
        N=N,
        kB=kB,
        m=m,
        L=L,
        V=V,
        temperatures=temperatures
    )

    plot_pressure(
        N=N,
        kB=kB,
        m=m,
        L=L,
        V=V,
        temperatures=temperatures
    )

    print("All tasks complete!!!")