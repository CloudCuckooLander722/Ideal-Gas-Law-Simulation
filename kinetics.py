import numpy as np


N = 1000          
kB = 1.3806e-23  
T = 300           
m = 6.64e-26     
L = 10

def initialize_system(N, L, kB, T, m):
    positional_vector = np.random.uniform(low=0.0, high=L, size=(N, 3))
    sigma = np.sqrt(kB*T/m) 
    velocity_init = np.random.normal(loc=0.0, scale=sigma, size=(N, 3))

    average_velocity = np.mean(velocity_init, axis=0)

    velocity_init = velocity_init - average_velocity

    T_current = (m * np.sum(velocity_init**2)) / (3 * N * kB)
    # 2. FIXED: Added np.sqrt()
    lambda_factor = np.sqrt(T / T_current)

    velocity_vector = velocity_init * lambda_factor

    return positional_vector, velocity_vector

def compute_momentum(positional_vector, velocity_vector, m): #rewrite
    total_delta_p = 0.0
    for axis in range(3):
        particles_hit = np.logical_or(
            positional_vector[:, axis] <= 0,
            positional_vector[:, axis] >= 10
        )
        if np.any(particles_hit):
            v_perp = np.abs(velocity_vector[particles_hit, axis])
            p_wall = 2 * m * v_perp
            total_delta_p += np.sum(p_wall)
            velocity_vector[particles_hit, axis] *= -1.0
            positional_vector[particles_hit, axis] = (2 * L) - positional_vector[particles_hit, axis]
            
    return total_delta_p


def compute_pressure(N, L, kB, T, m, steps=2000):
    
    positional_vector, velocity_vector = initialize_system(
        N=N, L=L, kB=kB, T=T, m=m
    )
    
    
    safety_factor = 0.01 
    max_speed = np.max(np.linalg.norm(velocity_vector, axis=1))
    dt = (safety_factor * L) / max_speed
    
    
    accumulated_delta_p = 0.0
    
    
    accumulated_delta_p, _ = update_positions(steps=steps, positional_vector=positional_vector, velocity_vector=velocity_vector, m=m, dt=dt)
    
    total_time = steps * dt
    surface_area = 6 * (L**2)
    
   
    return accumulated_delta_p / (total_time * surface_area)

def update_positions(steps, positional_vector, velocity_vector, m, dt):
    accumulated_delta_p = 0.0
    for _ in range(steps):
        
        positional_vector += velocity_vector * dt
        
        
        step_delta_p = compute_momentum(
            positional_vector=positional_vector, velocity_vector=velocity_vector, m=m
        )
       
        accumulated_delta_p += step_delta_p
    
    return accumulated_delta_p, positional_vector