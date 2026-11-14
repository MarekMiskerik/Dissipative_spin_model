import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import sympy as sp
import pandas as pd


def solve_and_plt_ODR(a, c, steps, t_max, h = 1):

    def f(t, v):
        g = np.zeros_like(v)
        g[0] = - h  * v[1]  + c * v[0] * v[2]
        g[1] = h * v[0] + 2 * a * v[0] * v[2] + c * v[1] * v[2]
        g[2] = - 2 * a * v[0] * v[1] - c * (v[0] ** 2 + v[1] ** 2)
        return g

    solution = spi.solve_ivp(f, [0, t_max], [1, 0, 0], method='RK45',  t_eval=np.linspace(0, t_max, steps))

    #J2 = solution.y[0] ** 2 + solution.y[1] ** 2 + solution.y[2] ** 2
    #print(solution.y[0][-1], solution.y[1][-1], solution.y[2][-1])

    #print(J2)
    
    #print(solution.y[2])

    plt.figure(figsize=(8, 5))

    plt.plot(solution.t, solution.y[0], label=r'$J_1$')
    plt.plot(solution.t, solution.y[1], label=r'$J_2$')
    plt.plot(solution.t, solution.y[2], label=r'$J_3$')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.show()


    v_dot = f(solution.t, solution.y)

    plt.figure(figsize=(8, 5))

    plt.plot(solution.y[0], v_dot[0], label=r'$J_1$ vs $\dot{J}_1$')
    plt.plot(solution.y[1], v_dot[1], label=r'$J_2$ vs $\dot{J}_2$')
    plt.plot(solution.y[2], v_dot[2], label=r'$J_3$ vs $\dot{J}_3$')

    plt.title('1D Phase Space: Derivative vs. Solution')
    plt.xlabel(r'Solution Component ($J_i$)')
    plt.ylabel(r'Time Derivative ($\dot{J}_i$)')
    plt.legend()
    plt.grid(True)

    plt.show()

# Find all fixed points symbolically but need to add a filter to remove the unphysical ones
# historically used to find the fixed points numerically as well, but too slow (see the commented out code below)
def sp_find_fixed_points(a, c,  h = 1):
    sp_j1, sp_j2, sp_j3 = sp.symbols('j1 j2 j3')
    sp_h, sp_a, sp_c = sp.symbols('h a c')

    eq1 = - sp_h * sp_j2 + sp_c * sp_j1 * sp_j3
    eq2 = sp_h * sp_j1 + 2 * sp_a * sp_j1 * sp_j3 + sp_c * sp_j2 * sp_j3
    eq3 = - 2 * sp_a * sp_j1 * sp_j2 - sp_c * (sp_j1 ** 2 + sp_j2 ** 2)
    eq4 = sp_j1 ** 2 + sp_j2 ** 2 + sp_j3 ** 2 - 1

    sp_fixed_points = sp.solve([eq1, eq2, eq3, eq4], (sp_j1, sp_j2, sp_j3))

    '''replacements = {sp_h: h, sp_a: a, sp_c: c}
    fixed_points = []
    for point in sp_fixed_points:
        j1 = point[0].subs(replacements).evalf()
        j2 = point[1].subs(replacements).evalf()
        j3 = point[2].subs(replacements).evalf()
        fixed_points.append((j1, j2, j3))

    real_fixed_points = []
    for point in fixed_points:
        j1_val = complex(point[0])
        j2_val = complex(point[1])
        j3_val = complex(point[2])
    
        if abs(np.imag(j3_val)) < 1e-10 and abs(np.imag(j2_val)) < 1e-10 and abs(np.imag(j1_val)) < 1e-10:
            real_fixed_points.append([np.real(j1_val), np.real(j2_val), np.real(j3_val)])'''

    return sp_fixed_points #, real_fixed_points

# finds the fixed points numerically, also returns the real ones only
# written entirely by Claude, same results as the symbolic version above but faster
def find_fixed_points(a, c, h = 1, tol=1e-12):
    pts = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]           # always fixed points

    if abs(h) > tol:
        # j3 solves c^2 j3^2 + 2 a h j3 + h^2 = 0 ; then j2 = k j1, k = c j3 / h
        if abs(c) > tol:
            D = a*a - c*c
            j3s = set() if D < -tol else {h*(-a + s)/c**2 for s in (np.sqrt(max(D, 0.0)),
                                                                         -np.sqrt(max(D, 0.0)))}
        elif abs(a) > tol:
            j3s = {-h / (2*a)}
        else:
            j3s = set()
        for j3 in j3s:
            if abs(j3) <= 1 + tol:
                k = c * j3 / h
                j1 = np.sqrt(max(1 - j3*j3, 0.0) / (1 + k*k))
                pts += [(j1, k*j1, j3), (-j1, -k*j1, j3)]
    else:
        # h = 0: j3 = 0 and a*sin(2*phi) = -c on the unit circle
        if abs(a) > tol and abs(c) <= abs(a) + tol:
            t = np.arcsin(np.clip(-c / a, -1, 1))
            for phi in {t/2, (np.pi - t)/2}:
                for s in (1, -1):
                    pts.append((s*np.cos(phi), s*np.sin(phi), 0.0))

    return np.unique(np.round(np.array(pts), 14), axis=0)   # shape (n, 3), real points only


# finds the eigenvalues of the Jacobian at a given fixed point
# we remove the redundant zero eigenvalue. can be done elsewhere depending on application
def Jacobian_eigenvalues(point, a, c, h = 1):
    sp_j1, sp_j2, sp_j3 = sp.symbols('j1 j2 j3')
    sp_h, sp_a, sp_c = sp.symbols('h a c')

    eq1 = - sp_h * sp_j2 + sp_c * sp_j1 * sp_j3
    eq2 = sp_h * sp_j1 + 2 * sp_a * sp_j1 * sp_j3 + sp_c * sp_j2 * sp_j3
    eq3 = - 2 * sp_a * sp_j1 * sp_j2 - sp_c * (sp_j1 ** 2 + sp_j2 ** 2)

    J = np.array([[sp.diff(eq1, var) for var in (sp_j1, sp_j2, sp_j3)],
                  [sp.diff(eq2, var) for var in (sp_j1, sp_j2, sp_j3)],
                  [sp.diff(eq3, var) for var in (sp_j1, sp_j2, sp_j3)]])

    replacements = {sp_h: h, sp_a: a, sp_c: c, sp_j1: point[0], sp_j2: point[1], sp_j3: point[2]}
    J_numeric = np.array([[entry.subs(replacements).evalf() for entry in row] for row in J], dtype=float)

    eigenvalues = np.linalg.eigvals(J_numeric)
    eigenvalues  = np.delete(eigenvalues, np.where(np.isclose(eigenvalues, 0, atol=1e-12)))
    return eigenvalues


# fucntion to ontain analytically expression for the eigenvalues of the Jacobian at a given fixed point
def sp_Jacobian_eigenvalues(a_num, c_num, h_num=1):
    h, a, c = sp.symbols('h a c')
    a_c = (h_num**2 + c_num**2) / (2*h_num)

    x3_p = -h / c**2 * (a + sp.sqrt(a**2 - c**2))
    x3_m = -h / c**2 * (a - sp.sqrt(a**2 - c**2))

    x1_p = h * sp.sqrt((1 - x3_p**2) / (h**2 + c**2 * x3_p**2))
    x1_m = h * sp.sqrt((1 - x3_m**2) / (h**2 + c**2 * x3_m**2))

    x2_p = c / h * x1_p * x3_p
    x2_m = c / h * x1_m * x3_m

    # all candidate fixed points, keyed by name
    candidates = {
        'S_pole': [0, 0, -1],
        'N_pole': [0, 0, 1],
        'x_pp':   [x1_p, x2_p, x3_p],
        'x_pm':   [-x1_p, -x2_p, x3_p],
        'x_mp':   [x1_m, x2_m, x3_m],
        'x_mm':   [-x1_m, -x2_m, x3_m],
    }

    j1, j2, j3 = sp.symbols('j1 j2 j3')

    # Define the system of equations
    eq1 = -h * j2 + c * j1 * j3
    eq2 = h * j1 + 2 * a * j1 * j3 + c * j2 * j3
    eq3 = -2 * a * j1 * j2 - c * (j1**2 + j2**2)

    system = sp.Matrix([eq1, eq2, eq3])
    state_vars = sp.Matrix([j1, j2, j3])
    J = system.jacobian(state_vars)

    def eig_entry(point):
        """Jacobian eigenvalues at one fixed point, simplified, as a table row."""
        J_p = J.subs({j1: point[0], j2: point[1], j3: point[2]})
        evs = [sp.simplify(ev) for ev in J_p.eigenvals(multiple=True)]
        return {
            'point': tuple(sp.simplify(x) for x in point),
            'lambda1': evs[0],
            'lambda2': evs[1],
            'lambda3': evs[2],
        }

    # Same existence logic as before, just choosing which points to include
    names = ['S_pole', 'N_pole']
    if h_num > c_num:
        if a_num > a_c:
            names += ['x_mp', 'x_mm']
    else:
        if a_num < c_num:
            pass
        elif a_num < a_c:
            names += ['x_mp', 'x_mm', 'x_pp', 'x_pm']
        else:
            names += ['x_mp', 'x_mm']

    results = {name: eig_entry(candidates[name]) for name in names}
    return pd.DataFrame.from_dict(results, orient='index')


# classifies the fixed point based on the eigenvalues of the Jacobian
# maybe can already pass the eigenvalues to avoid recomputing them
def fixed_point_character(point, a, c):
    #eigenvalues  = np.delete(eigenvalues, np.where(np.isclose(eigenvalues, 0, atol=1e-10)))
    eigenvalues = Jacobian_eigenvalues(point, a, c)   
    if np.all(np.real(eigenvalues) < 0):
        return "Stable"
    elif np.all(np.real(eigenvalues) > 0):
        return "Unstable"
    elif len(eigenvalues) < 2:
        return "Non-hyperbolic"
    else:
        return "Saddle"

# plots the eigenvalues of the Jacobian in a complex plain for differnent values of a
# put list of a values as argument 
def plot_fixed_points(a, c, h = 1):
    a_list, ev_list = [], []
    for a_val in a:
        fixed_points = find_fixed_points(h, a_val, c)
        for point in fixed_points:
            eigenvalues = Jacobian_eigenvalues(point, a=a_val, c=c)
            #eigenvalues = np.delete(eigenvalues, np.isclose(eigenvalues, 0, atol=1e-10))
            for eigenvalue in eigenvalues:
                ev_list.append(eigenvalue)
                a_list.append(a_val)
    a_list, ev_list = np.array(a_list), np.array(ev_list)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(ev_list.real, ev_list.imag, c=a_list, cmap="viridis", s=8)
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    plt.colorbar(sc, label="$a$")
    plt.xlabel(r"Re $\lambda$")
    plt.ylabel(r"Im $\lambda$")
    plt.show()


# Compute the eigenvalues of the Liouvillian correcpoing to a given fixed point using the Jacobian eigenlvaues 
def liouvillian_eigenvalues(J_eigenvalues, n_max, saddle):
    '''if saddle:
        if J_eigenvalues[0] > 0 or J_eigenvalues[1] > 0:
            return np.array([0])
        else:
            s = [n1 * J_eigenvalues[0] + n2 * J_eigenvalues[1] for n1 in range(n_max + 1) for n2 in range(n_max + 1 - n1)]
            return np.array(s)
    else:
        if J_eigenvalues[0] > 0 and J_eigenvalues[1] > 0:
            return np.array([0])
        else:
            s = [n1 * J_eigenvalues[0] + n2 * J_eigenvalues[1] for n1 in range(n_max + 1) for n2 in range(n_max + 1 - n1)]
            return np.array(s)'''
    s = [n1 * J_eigenvalues[0] + n2 * J_eigenvalues[1] for n1 in range(n_max + 1) for n2 in range(n_max + 1 - n1)]
    return np.array(s)

# another fucntion to convenietnly print some info about the fixed points, their Jacobian eigenvalues or their character
def print_fixed_points_info(fixed_points, a, c):
    for point in fixed_points:
        eigenvalues = Jacobian_eigenvalues(point, a, c)
        print(f"Fixed point: {point}")
        print(f"Jacobian Eigenvalues: {eigenvalues}")
        print(f"Character of the fixed point: {fixed_point_character(point, a, c)}")
        #print("Liouvillian Eigenvalues:")
        #print(liouvillian_eigenvalues(eigenvalues, 5))
        print("\n")



c = 2

h = 1
a_c = (h**2 + c**2)/(2*h)

a = c - 0.001

#solve_and_plt_ODR(a, c, steps=2000, t_max=200)

#df = sp_Jacobian_eigenvalues(a_num=a, c_num=c)
#print(eigenvalues_to_latex(df, filename='eigenvalues.tex'))


# block below computes and plots the Liouvillian eigenvalues corresponding fixed points using the fucntions 'louvillian_eigenvalues'
fixed_points = find_fixed_points(a=a, c=c)
print_fixed_points_info(fixed_points, a, c)
'''N_val = 6

all_L_evals = []  

plt.figure(figsize=(8, 8))

for i, point in enumerate(fixed_points):
    J_evals = Jacobian_eigenvalues(point, a=a, c=c)
    
    current_L_evals = liouvillian_eigenvalues(J_evals, N_val, saddle=False)
    all_L_evals.append(current_L_evals)
    
    # Plot inside the loop to get a unique color per fixed point
    plt.scatter(current_L_evals.real, current_L_evals.imag, 
                marker='o', alpha=0.5)#, label=f'Fixed Point {i+1}')

# Keep the concatenation in case you need all_L_evals later in your script
all_L_evals = np.concatenate(all_L_evals)

# Add axes lines through the origin
plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=0, color='k', linestyle='-', linewidth=1)

#plt.title('Liouvillian Eigenvalues - All Fixed Points')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True, linestyle='--', alpha=0.6)

# Display the legend to show the color mapping
plt.legend()

plt.gca().set_aspect('equal', adjustable='datalim')

plt.show()'''


# block below plots fixed points in spherical coordinates (theta, phi) for a range of a values, with color indicating the value of a
#aa = np.linspace(0, 10, 20) 
#plot_fixed_points(aa, c)

# converts cartesian coordinates on the Bloch sphere to spherical coordinates (theta, phi)
def cartesian_to_spherical(j1, j2, j3):

    theta = np.arccos(np.clip(j3, -1.0, 1.0))
    phi = np.arctan2(j2, j1)
    
    return theta, phi

'''
aa = np.linspace(0, 2.3, 10)
phi_plot = []
theta_plot = []
a_colors = []
for a_val in aa:
    fixed_points = find_fixed_points(a_val, c) # (n,3)
    
    for point in fixed_points[:]:
        j1_val, j2_val, j3_val = point[0], point[1], point[2]
        
        theta, phi = cartesian_to_spherical(j1_val, j2_val, j3_val)
        
        if True:  #phi > -1e-5: 
            phi_plot.append(phi)
            theta_plot.append(theta)
            a_colors.append(a_val) 

plt.figure(figsize=(7, 5))

scatter = plt.scatter(phi_plot, theta_plot, c=a_colors, cmap='Reds', s=40, edgecolor='black', linewidth=0.5, zorder=3)

plt.xlim(-np.pi, np.pi + 0.2)
plt.ylim(0, np.pi + 0.2)

# adjusting the ticks to show multiples of pi
#plt.xticks([0, np.pi/2, np.pi], [r'$0$', r'$\pi/2$', r'$\pi$'])
#plt.yticks([0, np.pi/2, np.pi], [r'$0$', r'$\pi/2$', r'$\pi$'])

plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')

# color bar
cbar = plt.colorbar(scatter)
cbar.set_label(r'Coupling parameter $a$')

plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.show()



fig = plt.figure(figsize=(9, 7))

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# X = phi, Y = theta, Z = a
# We can keep the c=a_colors mapping to give the points a nice depth gradient
scatter = ax.scatter(phi_plot, theta_plot, a_colors, 
                     c=a_colors, cmap='Reds', s=40, edgecolor='black', linewidth=0.5)

# Strictly format the X and Y axes to match the physical bounds of the Bloch sphere
ax.set_xlim(0, np.pi)
ax.set_ylim(0, np.pi)

ax.set_xticks([0, np.pi/2, np.pi])
ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

ax.set_yticks([0, np.pi/2, np.pi])
ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

ax.set_xlabel(r'Azimuthal angle $\phi$')
ax.set_ylabel(r'Polar angle $\theta$')
ax.set_zlabel(r'Coupling parameter $a$')

# Adjust the initial viewing angle (elevation and azimuth) for best visibility
ax.view_init(elev=20, azim=-45)

plt.tight_layout()
plt.show()'''