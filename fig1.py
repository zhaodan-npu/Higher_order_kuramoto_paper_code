import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


K1 = 0.8
K2 = 8.0


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
})


def kuramoto_ode(t, r):
    return -r + (K1/2)*(r - r**3) + (K2/2)*(r**3 - r**5)


r_initial = np.linspace(0, 1, 1000)


t_span = [0, 100]


r_final = []


for r0 in r_initial:
    sol = solve_ivp(kuramoto_ode, t_span, [r0], method='RK45', rtol=1e-6)
    r_final.append(sol.y[0, -1])


r_final = np.array(r_final)
transition_index = np.where(r_final > 0.5)[0][0]
transition_r_initial = r_initial[transition_index]


r_initial_before = r_initial[:transition_index]
r_final_before = r_final[:transition_index]

r_initial_after = r_initial[transition_index:]
r_final_after = r_final[transition_index:]


plt.figure(figsize=(8, 5))
plt.plot(r_initial_before, r_final_before, '.', color='blue', markersize=3, label='Steady state (before transition)')
plt.plot(r_initial_after, r_final_after, '.', color='red', markersize=3, label='Steady state (after transition)')


plt.axvline(transition_r_initial, color='green', linestyle='--', linewidth=2,
            label=f'Transition at r={transition_r_initial:.2f}')


plt.xlabel('$r_0$', fontsize=18)
plt.ylabel('Steady state', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('transition_plot.pdf', bbox_inches='tight')

plt.show()
