import numpy as np


def neuron_constant_current( R, C, tau_m, u_rest, u_threshold, u_reset, I, dt, t_max):
    t_values = np.arange(0, t_max, dt)
    u_values = np.zeros_like(t_values)

    u = u_rest
    for i, t in enumerate(t_values):
        # Update membrane potential using Euler's method
        du_dt = (u_rest - u + R * I) / tau_m
        u += du_dt * dt

        # Check for firing condition
        if u >= u_threshold:
            u_values[i] = u_threshold
            u = u_reset  # Reset potential

        u_values[i] = u

    return t_values, u_values