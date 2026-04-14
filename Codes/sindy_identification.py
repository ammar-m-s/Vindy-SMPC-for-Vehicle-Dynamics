"""
sindy_identification.py  -  Stage 2: SINDy identification of vehicle dynamics.

Loads sindy_data.npz (from data_gen_sindy.py) and identifies the 25 coefficients
c1-c25 of the 3-DOF bicycle model (Equation 16) using Sequential Thresholded
Least Squares (STLS), following the paper exactly.

Model being identified (Eq. 16):
    vx_dot    = c1*omega*vy + c2*Tr
                + (c3*af + c4*af^3 + c5*af^5 + c6*af^7)*sin(delta)
                + c7*vx + c8*vx^2

    vy_dot    = c9*omega*vx
                + (c10*af + c11*af^3 + c12*af^5 + c13*af^7)*cos(delta)
                + c14*ar + c15*ar^3 + c16*ar^5 + c17*ar^7

    omega_dot = (c18*af + c19*af^3 + c20*af^5 + c21*af^7)*cos(delta)
                + c22*ar + c23*ar^3 + c24*ar^5 + c25*ar^7

Outputs:
    sindy_coefficients.npz      - identified c1-c25
    sindy_plot_parity.png       - predicted vs actual scatter
    sindy_plot_coefficients.png - bar chart of coefficients
    sindy_plot_timeseries.png   - time-series fit comparison
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

_save_only = not sys.stdout.isatty()

# ============================================================
#  1. LOAD DATA
# ============================================================
data = np.load('sindy_data.npz')
X_sindy    = data['X_sindy']     # (N, 5): [vx, vy, omega, alpha_f, alpha_r]
Xdot_sindy = data['Xdot_sindy'] # (N, 3): [vx_dot, vy_dot, omega_dot]
U_sindy    = data['U_sindy']     # (N, 2): [delta, Tr]

N = X_sindy.shape[0]
print(f"Loaded {N} samples from sindy_data.npz")

# Filter out corrupt samples (vx ~ 0 from lap-boundary logging artifact)
valid = X_sindy[:, 0] > 1.0
n_removed = (~valid).sum()
if n_removed:
    print(f"  Removed {n_removed} corrupt sample(s) with vx < 1 m/s")
    X_sindy    = X_sindy[valid]
    Xdot_sindy = Xdot_sindy[valid]
    U_sindy    = U_sindy[valid]

N = X_sindy.shape[0]
print(f"  Using {N} samples")
print(f"  State ranges:")
labels_X = ['vx [m/s]', 'vy [m/s]', 'omega [rad/s]', 'alpha_f [rad]', 'alpha_r [rad]']
for i, lbl in enumerate(labels_X):
    print(f"    {lbl:<18}  [{X_sindy[:,i].min():+.4f}, {X_sindy[:,i].max():+.4f}]")

# Unpack columns
vx    = X_sindy[:, 0]
vy    = X_sindy[:, 1]
omega = X_sindy[:, 2]
af    = X_sindy[:, 3]   # front slip angle [rad]
ar    = X_sindy[:, 4]   # rear  slip angle [rad]
delta = U_sindy[:, 0]   # front steering angle [rad]
Tr    = U_sindy[:, 1]   # rear axle torque [Nm]

vx_dot    = Xdot_sindy[:, 0]
vy_dot    = Xdot_sindy[:, 1]
omega_dot = Xdot_sindy[:, 2]

# ============================================================
#  2. BUILD PHYSICS-INFORMED LIBRARY MATRICES  Theta(X, U)
#     Each equation has its own library - only terms with known
#     physical meaning are included (odd polynomials of slip angles
#     justified by the odd symmetry of Pacejka lateral forces).
# ============================================================

# --- vx_dot library  (8 columns  ->  c1 ... c8) ---
Theta_vx = np.column_stack([
    omega * vy,              # c1  Coriolis coupling
    Tr,                      # c2  drive torque  (~ Tr / (M*R_wheel) after identification)
    af     * np.sin(delta),  # c3  front lat force, linear   × sin(delta)
    af**3  * np.sin(delta),  # c4  front lat force, cubic    × sin(delta)
    af**5  * np.sin(delta),  # c5  front lat force, quintic  × sin(delta)
    af**7  * np.sin(delta),  # c6  front lat force, septic   × sin(delta)
    np.ones(N),              # c7  constant  (rolling resistance -Cr*g)
    vx**2,                   # c8  aero drag  (~ -0.5rhoCdAf/M)
    # NOTE: the paper writes c7*vx, but our vehicle uses Fres = Cr*M*g + 0.5*rho*Cd*Af*vx^2
    # whose vx_dot contribution is -Cr*g (constant) - 0.5*rho*Cd*Af/M*vx^2 (quadratic).
    # Replacing vx with 1 recovers R^2=1.000 and the exact expected coefficients.
])

# --- vy_dot library  (9 columns  ->  c9 ... c17) ---
Theta_vy = np.column_stack([
    omega * vx,              # c9  Coriolis coupling
    af     * np.cos(delta),  # c10 front lat force, linear   × cos(delta)
    af**3  * np.cos(delta),  # c11 front lat force, cubic    × cos(delta)
    af**5  * np.cos(delta),  # c12 front lat force, quintic  × cos(delta)
    af**7  * np.cos(delta),  # c13 front lat force, septic   × cos(delta)
    ar,                      # c14 rear lat force, linear
    ar**3,                   # c15 rear lat force, cubic
    ar**5,                   # c16 rear lat force, quintic
    ar**7,                   # c17 rear lat force, septic
])

# --- omega_dot library  (8 columns  ->  c18 ... c25) ---
Theta_om = np.column_stack([
    af     * np.cos(delta),  # c18 front yaw moment, linear   × cos(delta)
    af**3  * np.cos(delta),  # c19 front yaw moment, cubic    × cos(delta)
    af**5  * np.cos(delta),  # c20 front yaw moment, quintic  × cos(delta)
    af**7  * np.cos(delta),  # c21 front yaw moment, septic   × cos(delta)
    ar,                      # c22 rear yaw moment, linear
    ar**3,                   # c23 rear yaw moment, cubic
    ar**5,                   # c24 rear yaw moment, quintic
    ar**7,                   # c25 rear yaw moment, septic
])

# ============================================================
#  3. STLS ALGORITHM  (Algorithm 1 in paper, N_SINDy = 10)
# ============================================================

def stls(Theta, y, lam, n_iter=10):
    """
    Sequential Thresholded Least Squares.

    Solves:  phi = argmin  (1/2)||y - Theta @ phi||^2  +  lam||phi||_1
    via iterative hard thresholding on the LS solution.

    Normalization is applied column-wise before solving (so lam is
    scale-invariant) and reversed afterwards.

    Parameters
    ----------
    Theta  : (N, p) library matrix
    y      : (N,)   target time-derivative
    lam    : sparsity threshold in normalized space
    n_iter : number of thresholding passes (paper uses 10)

    Returns
    -------
    xi : (p,) coefficient vector (in original, un-normalized units)
    """
    # Column-wise normalization  (lambda must be < smallest expected |coeff_normalized|)
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Th = Theta / col_norms          # each column has unit L2-norm

    # Initial unconstrained least-squares
    xi, _, _, _ = np.linalg.lstsq(Th, y, rcond=None)

    # Iterative thresholding + re-fit
    for _ in range(n_iter):
        active = np.abs(xi) >= lam
        if active.sum() == 0:
            xi[:] = 0.0
            break
        xi_new = np.zeros_like(xi)
        xi_new[active], _, _, _ = np.linalg.lstsq(Th[:, active], y, rcond=None)
        xi = xi_new

    # Undo normalization: Theta_norm @ xi_norm = y  ->  xi = xi_norm / col_norms
    return xi / col_norms


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


# ============================================================
#  4. RUN STLS  +  AUTO-TUNE LAMBDA
# ============================================================

def run_stls_all(lam):
    c_vx  = stls(Theta_vx, vx_dot,    lam)
    c_vy  = stls(Theta_vy, vy_dot,    lam)
    c_om  = stls(Theta_om, omega_dot, lam)
    r2s   = [r_squared(vx_dot,    Theta_vx @ c_vx),
             r_squared(vy_dot,    Theta_vy @ c_vy),
             r_squared(omega_dot, Theta_om @ c_om)]
    return c_vx, c_vy, c_om, r2s


# First pass with paper's suggested starting point (small λ)
lambda_val  = 1e-3
c_vx, c_vy, c_om, r2s = run_stls_all(lambda_val)

print(f"\n=== STLS  (lambda={lambda_val:.1e},  N_iter=10) ===")
print(f"  R^2(vx_dot)    = {r2s[0]:.7f}")
print(f"  R^2(vy_dot)    = {r2s[1]:.7f}")
print(f"  R^2(omega_dot) = {r2s[2]:.7f}")

# Auto-tune if any R^2 < 0.999
if min(r2s) < 0.999:
    print(f"\n  min R^2 = {min(r2s):.6f} < 0.999 -- sweeping lambda ...")
    best_r2  = min(r2s)
    best_lam = lambda_val
    best_res = (c_vx, c_vy, c_om, r2s)

    for lam in np.logspace(-6, 0, 80):
        cx, cv, co, rs = run_stls_all(lam)
        if min(rs) > best_r2:
            best_r2  = min(rs)
            best_lam = lam
            best_res = (cx, cv, co, rs)

    lambda_val          = best_lam
    c_vx, c_vy, c_om, r2s = best_res
    print(f"  Best lambda = {lambda_val:.2e}  ->  min R^2 = {best_r2:.7f}")
    print(f"  R^2(vx_dot)    = {r2s[0]:.7f}")
    print(f"  R^2(vy_dot)    = {r2s[1]:.7f}")
    print(f"  R^2(omega_dot) = {r2s[2]:.7f}")

R2_vx, R2_vy, R2_omega = r2s

# ============================================================
#  5. COEFFICIENT TABLE
# ============================================================
labels = {
    'vx': [
        ('c1',  'omega*vy'),
        ('c2',  'Tr'),
        ('c3',  'af*sin(delta)'),
        ('c4',  'af^3*sin(delta)'),
        ('c5',  'af^5*sin(delta)'),
        ('c6',  'af^7*sin(delta)'),
        ('c7',  '1 (const)'),
        ('c8',  'vx^2'),
    ],
    'vy': [
        ('c9',  'omega*vx'),
        ('c10', 'af*cos(delta)'),
        ('c11', 'af^3*cos(delta)'),
        ('c12', 'af^5*cos(delta)'),
        ('c13', 'af^7*cos(delta)'),
        ('c14', 'ar'),
        ('c15', 'ar^3'),
        ('c16', 'ar^5'),
        ('c17', 'ar^7'),
    ],
    'om': [
        ('c18', 'af*cos(delta)'),
        ('c19', 'af^3*cos(delta)'),
        ('c20', 'af^5*cos(delta)'),
        ('c21', 'af^7*cos(delta)'),
        ('c22', 'ar'),
        ('c23', 'ar^3'),
        ('c24', 'ar^5'),
        ('c25', 'ar^7'),
    ],
}

print("\n" + "="*60)
print("IDENTIFIED COEFFICIENTS")
print("="*60)
for eq_name, coeff_vec, lbls in [
        ('vx_dot',    c_vx, labels['vx']),
        ('vy_dot',    c_vy, labels['vy']),
        ('omega_dot', c_om, labels['om']),
    ]:
    print(f"\n  {eq_name}:")
    for (cname, term), val in zip(lbls, coeff_vec):
        status = '  ← zeroed' if val == 0.0 else ''
        print(f"    {cname:<4}  {term:<18}  {val:+.6e}{status}")
print()

# ============================================================
#  6. PHYSICAL SANITY CHECK
# ============================================================
M       = 1412.0;  R_wheel = 0.325;  Cr   = 0.015;  g  = 9.81
rho     = 1.225;   Cd      = 0.3;    Af   = 2.2;    Jz = 1536.7
lf      = 1.015;   lr      = 1.895
B_pac   = 0.0885 * (180/np.pi)       # Pacejka B in rad⁻¹
C_pac   = 1.4;     D_pac   = 8311.0

# Linear Pacejka cornering stiffness: dF/dalpha|_{alpha=0} = D*C*B
Cf_lin  = D_pac * C_pac * B_pac      # [N/rad]

print("="*60)
print("PHYSICAL SANITY CHECK")
print("="*60)
checks = [
    ('c1',  c_vx[0],  +1.0,                        'Coriolis (expect +1)'),
    ('c2',  c_vx[1],  +1/(M*R_wheel),               'Torque/mass (expect +1/(M*Rw))'),
    ('c7',  c_vx[6],  -Cr*g,                        'Roll. resist. (expect -Cr*g)'),
    ('c8',  c_vx[7],  -0.5*rho*Cd*Af/M,             'Aero drag (expect -0.5rhoCdAf/M)'),
    ('c9',  c_vy[0],  -1.0,                         'Coriolis (expect -1)'),
    ('c10', c_vy[1],  +Cf_lin/M,                    'Front cornering stiffness/M'),
    ('c14', c_vy[5],  +Cf_lin/M,                    'Rear  cornering stiffness/M'),
    ('c18', c_om[0],  +Cf_lin*lf/Jz,                'Front moment arm/Jz'),
    ('c22', c_om[4],  -Cf_lin*lr/Jz,                'Rear  moment arm/Jz'),
]
print(f"  {'Coeff':<5}  {'Identified':>14}  {'Expected':>14}  {'Ratio':>8}  Description")
print(f"  {'-'*4}   {'-'*13}   {'-'*13}   {'-'*7}  {'-'*30}")
for cname, val, expected, desc in checks:
    ratio = val/expected if abs(expected) > 1e-12 else float('nan')
    print(f"  {cname:<5}  {val:+14.5e}  {expected:+14.5e}  {ratio:8.4f}  {desc}")

# ============================================================
#  7. SAVE
# ============================================================
c_all = np.concatenate([c_vx, c_vy, c_om])   # c1 ... c25 in order

np.savez('sindy_coefficients.npz',
         c_vx=c_vx, c_vy=c_vy, c_omega=c_om,
         c_all=c_all,
         R2_vx=np.array(R2_vx), R2_vy=np.array(R2_vy), R2_omega=np.array(R2_omega),
         lambda_val=np.array(lambda_val))
print(f"\nSaved sindy_coefficients.npz  (c_all shape: {c_all.shape})")

# ============================================================
#  8. PLOTS
# ============================================================

# --- Parity plots ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('SINDy: Predicted vs Actual Derivatives', fontsize=12)
plot_data = [
    (vx_dot,    Theta_vx @ c_vx,    R2_vx,    r'$\dot{v}_x$ [m/s^2]'),
    (vy_dot,    Theta_vy @ c_vy,    R2_vy,    r'$\dot{v}_y$ [m/s^2]'),
    (omega_dot, Theta_om @ c_om,    R2_omega, r'$\dot{\omega}$ [rad/s^2]'),
]
for ax, (y_true, y_pred, r2, lbl) in zip(axes, plot_data):
    ax.scatter(y_true, y_pred, s=1, alpha=0.2, color='steelblue', rasterized=True)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y=x')
    ax.set_xlabel(f'True {lbl}', fontsize=9)
    ax.set_ylabel(f'SINDy {lbl}', fontsize=9)
    ax.set_title(f'R^2 = {r2:.7f}', fontsize=10)
    ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('sindy_plot_parity.png', dpi=120, bbox_inches='tight')

# --- Coefficient bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Identified SINDy Coefficients  (grey = zeroed by STLS)', fontsize=11)
for ax, (coeff_vec, lbls, title) in zip(axes, [
        (c_vx, labels['vx'], r'$\dot{v}_x$  (c1-c8)'),
        (c_vy, labels['vy'], r'$\dot{v}_y$  (c9-c17)'),
        (c_om, labels['om'], r'$\dot{\omega}$  (c18-c25)'),
    ]):
    xlbls   = [f"{n}\n{t}" for n, t in lbls]
    bar_clr = ['steelblue' if v != 0 else 'lightgrey' for v in coeff_vec]
    ax.bar(xlbls, coeff_vec, color=bar_clr, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('sindy_plot_coefficients.png', dpi=120, bbox_inches='tight')

# --- Time-series comparison: pick a window with high slip angle activity ---
# Find the 1000-sample window with the largest variance in omega_dot (most dynamic)
window = 1000
n_windows = N - window
var_om = np.array([omega_dot[i:i+window].var() for i in range(0, n_windows, 50)])
best_start = np.argmax(var_om) * 50
idx_plot = np.arange(best_start, best_start + window)

fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
fig.suptitle(f'SINDy Model Time-series Fit  (most dynamic 1000-sample window, start={best_start})', fontsize=10)
pred_vx = Theta_vx @ c_vx
pred_vy = Theta_vy @ c_vy
pred_om = Theta_om @ c_om
ts_data = [
    (vx_dot[idx_plot],    pred_vx[idx_plot],    r'$\dot{v}_x$ [m/s^2]'),
    (vy_dot[idx_plot],    pred_vy[idx_plot],    r'$\dot{v}_y$ [m/s^2]'),
    (omega_dot[idx_plot], pred_om[idx_plot],    r'$\dot{\omega}$ [rad/s^2]'),
]
for ax, (y_true, y_pred, lbl) in zip(axes, ts_data):
    ax.plot(idx_plot, y_true, 'k-',  lw=1,   label='True')
    ax.plot(idx_plot, y_pred, 'r--', lw=0.8, label='SINDy')
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.4)
    if ax is axes[0]:
        ax.legend(loc='upper right', fontsize=9)
axes[-1].set_xlabel('Sample index')
plt.tight_layout()
plt.savefig('sindy_plot_timeseries.png', dpi=120, bbox_inches='tight')

print("Saved sindy_plot_parity.png, sindy_plot_coefficients.png, sindy_plot_timeseries.png")

if not _save_only:
    plt.show()
