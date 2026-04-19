"""
sindy_identification.py  -  Stage 2: SINDy identification of vehicle dynamics.

Loads sindy_data.npz (from data_gen_sindy.py) and identifies the 25 coefficients
c1-c25 of the 3-DOF bicycle model (Equation 16) using Sequential Thresholded
Least Squares (STLS).

This version assumes the dataset already contains normalized, dimensionless
state/input/derivative arrays, along with scale vectors that map between
normalized and physical units.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

_save_only = not sys.stdout.isatty()

# ============================================================
#  1. LOAD DATA
# ============================================================
data = np.load('sindy_data_normalized.npz', allow_pickle=True)
X_sindy    = data['X_sindy']
Xdot_sindy = data['Xdot_sindy']
U_sindy    = data['U_sindy']

X_sindy_raw    = data['X_sindy_raw'] if 'X_sindy_raw' in data else None
Xdot_sindy_raw = data['Xdot_sindy_raw'] if 'Xdot_sindy_raw' in data else None
U_sindy_raw    = data['U_sindy_raw'] if 'U_sindy_raw' in data else None
x_scale        = data['x_scale']
xdot_scale     = data['xdot_scale']
u_scale        = data['u_scale']

N = X_sindy.shape[0]
print(f"Loaded {N} samples from sindy_data_normalized.npz")
print("Using normalized dimensionless arrays:")
print(f"  x_scale    = {x_scale}")
print(f"  xdot_scale = {xdot_scale}")
print(f"  u_scale    = {u_scale}")

valid = X_sindy[:, 0] > (1.0 / x_scale[0])
n_removed = (~valid).sum()
if n_removed:
    print(f"  Removed {n_removed} corrupt sample(s) with vx < 1 m/s")
    X_sindy    = X_sindy[valid]
    Xdot_sindy = Xdot_sindy[valid]
    U_sindy    = U_sindy[valid]
    if X_sindy_raw is not None:
        X_sindy_raw = X_sindy_raw[valid]
        Xdot_sindy_raw = Xdot_sindy_raw[valid]
        U_sindy_raw = U_sindy_raw[valid]

N = X_sindy.shape[0]
print(f"  Using {N} samples")
print("  Normalized state ranges:")
labels_X = ['vx [-]', 'vy [-]', 'omega [-]', r'alpha_f [-]', r'alpha_r [-]']
for i, lbl in enumerate(labels_X):
    print(f"    {lbl:<18}  [{X_sindy[:,i].min():+.4f}, {X_sindy[:,i].max():+.4f}]")

# Unpack normalized columns
vx_n    = X_sindy[:, 0]
vy_n    = X_sindy[:, 1]
omega_n = X_sindy[:, 2]
af_n    = X_sindy[:, 3]
ar_n    = X_sindy[:, 4]
delta_n = U_sindy[:, 0]
Tr_n    = U_sindy[:, 1]

vx_dot_n    = Xdot_sindy[:, 0]
vy_dot_n    = Xdot_sindy[:, 1]
omega_dot_n = Xdot_sindy[:, 2]

# Physical-unit views retained for reporting / optional sanity checking
if X_sindy_raw is None:
    X_sindy_raw = X_sindy * x_scale
    Xdot_sindy_raw = Xdot_sindy * xdot_scale
    U_sindy_raw = U_sindy * u_scale

vx    = X_sindy_raw[:, 0]
vy    = X_sindy_raw[:, 1]
omega = X_sindy_raw[:, 2]
af    = X_sindy_raw[:, 3]
ar    = X_sindy_raw[:, 4]
delta = U_sindy_raw[:, 0]
Tr    = U_sindy_raw[:, 1]

vx_dot    = Xdot_sindy_raw[:, 0]
vy_dot    = Xdot_sindy_raw[:, 1]
omega_dot = Xdot_sindy_raw[:, 2]

# ============================================================
#  2. BUILD PHYSICS-INFORMED LIBRARY MATRICES IN NORMALIZED UNITS
# ============================================================
Theta_vx = np.column_stack([
    omega_n * vy_n,
    Tr_n,
    af_n    * np.sin(delta_n),
    af_n**3 * np.sin(delta_n),
    af_n**5 * np.sin(delta_n),
    af_n**7 * np.sin(delta_n),
    np.ones(N),
    vx_n**2,
])

Theta_vy = np.column_stack([
    omega_n * vx_n,
    af_n    * np.cos(delta_n),
    af_n**3 * np.cos(delta_n),
    af_n**5 * np.cos(delta_n),
    af_n**7 * np.cos(delta_n),
    ar_n,
    ar_n**3,
    ar_n**5,
    ar_n**7,
])

Theta_om = np.column_stack([
    af_n    * np.cos(delta_n),
    af_n**3 * np.cos(delta_n),
    af_n**5 * np.cos(delta_n),
    af_n**7 * np.cos(delta_n),
    ar_n,
    ar_n**3,
    ar_n**5,
    ar_n**7,
])

# ============================================================
#  3. STLS ALGORITHM
# ============================================================

def stls(Theta, y, lam, n_iter=10):
    """Sequential Thresholded Least Squares on a normalized library."""
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Th = Theta / col_norms

    xi, _, _, _ = np.linalg.lstsq(Th, y, rcond=None)

    for _ in range(n_iter):
        active = np.abs(xi) >= lam
        if active.sum() == 0:
            xi[:] = 0.0
            break
        xi_new = np.zeros_like(xi)
        xi_new[active], _, _, _ = np.linalg.lstsq(Th[:, active], y, rcond=None)
        xi = xi_new

    return xi / col_norms


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def convert_coeffs_to_physical_units(c_norm, eq_name, x_scale, xdot_scale, u_scale):
    """Map coefficients from normalized model back to physical SI-unit coefficients."""
    s_vx, s_vy, s_om, s_af, s_ar = x_scale
    s_vx_dot, s_vy_dot, s_om_dot = xdot_scale
    s_delta, s_Tr = u_scale

    if eq_name == 'vx':
        theta_scales = np.array([
            s_om * s_vy,
            s_Tr,
            s_af,
            s_af**3,
            s_af**5,
            s_af**7,
            1.0,
            s_vx**2,
        ])
        y_scale = s_vx_dot
    elif eq_name == 'vy':
        theta_scales = np.array([
            s_om * s_vx,
            s_af,
            s_af**3,
            s_af**5,
            s_af**7,
            s_ar,
            s_ar**3,
            s_ar**5,
            s_ar**7,
        ])
        y_scale = s_vy_dot
    elif eq_name == 'om':
        theta_scales = np.array([
            s_af,
            s_af**3,
            s_af**5,
            s_af**7,
            s_ar,
            s_ar**3,
            s_ar**5,
            s_ar**7,
        ])
        y_scale = s_om_dot
    else:
        raise ValueError(f'Unknown equation name: {eq_name}')

    return c_norm * y_scale / theta_scales


# ============================================================
#  4. RUN STLS  +  AUTO-TUNE LAMBDA
# ============================================================

def run_stls_all(lam):
    c_vx_n  = stls(Theta_vx, vx_dot_n,    lam)
    c_vy_n  = stls(Theta_vy, vy_dot_n,    lam)
    c_om_n  = stls(Theta_om, omega_dot_n, lam)
    r2s   = [
        r_squared(vx_dot_n,    Theta_vx @ c_vx_n),
        r_squared(vy_dot_n,    Theta_vy @ c_vy_n),
        r_squared(omega_dot_n, Theta_om @ c_om_n),
    ]
    return c_vx_n, c_vy_n, c_om_n, r2s


lambda_val  = 1e-3
c_vx_n, c_vy_n, c_om_n, r2s = run_stls_all(lambda_val)

print(f"\n=== STLS on normalized data  (lambda={lambda_val:.1e},  N_iter=10) ===")
print(f"  R^2(vx_dot_n)    = {r2s[0]:.7f}")
print(f"  R^2(vy_dot_n)    = {r2s[1]:.7f}")
print(f"  R^2(omega_dot_n) = {r2s[2]:.7f}")

if min(r2s) < 0.999:
    print(f"\n  min R^2 = {min(r2s):.6f} < 0.999 -- sweeping lambda ...")
    best_r2  = min(r2s)
    best_lam = lambda_val
    best_res = (c_vx_n, c_vy_n, c_om_n, r2s)

    for lam in np.logspace(-6, 0, 80):
        cx, cv, co, rs = run_stls_all(lam)
        if min(rs) > best_r2:
            best_r2  = min(rs)
            best_lam = lam
            best_res = (cx, cv, co, rs)

    lambda_val = best_lam
    c_vx_n, c_vy_n, c_om_n, r2s = best_res
    print(f"  Best lambda = {lambda_val:.2e}  ->  min R^2 = {best_r2:.7f}")
    print(f"  R^2(vx_dot_n)    = {r2s[0]:.7f}")
    print(f"  R^2(vy_dot_n)    = {r2s[1]:.7f}")
    print(f"  R^2(omega_dot_n) = {r2s[2]:.7f}")

R2_vx, R2_vy, R2_omega = r2s

c_vx = convert_coeffs_to_physical_units(c_vx_n, 'vx', x_scale, xdot_scale, u_scale)
c_vy = convert_coeffs_to_physical_units(c_vy_n, 'vy', x_scale, xdot_scale, u_scale)
c_om = convert_coeffs_to_physical_units(c_om_n, 'om', x_scale, xdot_scale, u_scale)

# ============================================================
#  5. COEFFICIENT TABLES
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
print("IDENTIFIED COEFFICIENTS (NORMALIZED MODEL)")
print("="*60)
for eq_name, coeff_vec, lbls in [
        ('vx_dot_n',    c_vx_n, labels['vx']),
        ('vy_dot_n',    c_vy_n, labels['vy']),
        ('omega_dot_n', c_om_n, labels['om']),
    ]:
    print(f"\n  {eq_name}:")
    for (cname, term), val in zip(lbls, coeff_vec):
        status = '  ← zeroed' if val == 0.0 else ''
        print(f"    {cname:<4}  {term:<18}  {val:+.6e}{status}")

print("\n" + "="*60)
print("IDENTIFIED COEFFICIENTS (PHYSICAL SI UNITS)")
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
B_pac   = 0.0885 * (180/np.pi)
C_pac   = 1.4;     D_pac   = 8311.0
Cf_lin  = D_pac * C_pac * B_pac

print("="*60)
print("PHYSICAL SANITY CHECK")
print("="*60)
checks = [
    ('c1',  c_vx[0],  +1.0,                        'Coriolis (expect +1)'),
    ('c2',  c_vx[1],  +1/(M*R_wheel),              'Torque/mass (expect +1/(M*Rw))'),
    ('c7',  c_vx[6],  -Cr*g,                       'Roll. resist. (expect -Cr*g)'),
    ('c8',  c_vx[7],  -0.5*rho*Cd*Af/M,            'Aero drag (expect -0.5rhoCdAf/M)'),
    ('c9',  c_vy[0],  -1.0,                        'Coriolis (expect -1)'),
    ('c10', c_vy[1],  +Cf_lin/M,                   'Front cornering stiffness/M'),
    ('c14', c_vy[5],  +Cf_lin/M,                   'Rear  cornering stiffness/M'),
    ('c18', c_om[0],  +Cf_lin*lf/Jz,               'Front moment arm/Jz'),
    ('c22', c_om[4],  -Cf_lin*lr/Jz,               'Rear  moment arm/Jz'),
]
print(f"  {'Coeff':<5}  {'Identified':>14}  {'Expected':>14}  {'Ratio':>8}  Description")
print(f"  {'-'*4}   {'-'*13}   {'-'*13}   {'-'*7}  {'-'*30}")
for cname, val, expected, desc in checks:
    ratio = val/expected if abs(expected) > 1e-12 else float('nan')
    print(f"  {cname:<5}  {val:+14.5e}  {expected:+14.5e}  {ratio:8.4f}  {desc}")

# ============================================================
#  7. SAVE
# ============================================================
c_all_n = np.concatenate([c_vx_n, c_vy_n, c_om_n])
c_all   = np.concatenate([c_vx, c_vy, c_om])

np.savez(
    'sindy_coefficients.npz',
    c_vx=c_vx, c_vy=c_vy, c_omega=c_om,
    c_vx_normalized=c_vx_n, c_vy_normalized=c_vy_n, c_omega_normalized=c_om_n,
    c_all=c_all, c_all_normalized=c_all_n,
    R2_vx=np.array(R2_vx), R2_vy=np.array(R2_vy), R2_omega=np.array(R2_omega),
    lambda_val=np.array(lambda_val),
    x_scale=x_scale, xdot_scale=xdot_scale, u_scale=u_scale,
)
print(f"\nSaved sindy_coefficients.npz  (c_all shape: {c_all.shape})")

# ============================================================
#  8. PLOTS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('SINDy: Predicted vs Actual Normalized Derivatives', fontsize=12)
plot_data = [
    (vx_dot_n,    Theta_vx @ c_vx_n,    R2_vx,    r'$\dot{v}_{x,n}$ [-]'),
    (vy_dot_n,    Theta_vy @ c_vy_n,    R2_vy,    r'$\dot{v}_{y,n}$ [-]'),
    (omega_dot_n, Theta_om @ c_om_n,    R2_omega, r'$\dot{\omega}_n$ [-]'),
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

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Identified SINDy Coefficients in Normalized Coordinates', fontsize=11)
for ax, (coeff_vec, lbls, title) in zip(axes, [
        (c_vx_n, labels['vx'], r'$\dot{v}_{x,n}$  (c1-c8)'),
        (c_vy_n, labels['vy'], r'$\dot{v}_{y,n}$  (c9-c17)'),
        (c_om_n, labels['om'], r'$\dot{\omega}_n$  (c18-c25)'),
    ]):
    xlbls   = [f"{n}\n{t}" for n, t in lbls]
    bar_clr = ['steelblue' if v != 0 else 'lightgrey' for v in coeff_vec]
    ax.bar(xlbls, coeff_vec, color=bar_clr, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Value [-]')
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('sindy_plot_coefficients.png', dpi=120, bbox_inches='tight')

window = 1000
n_windows = max(N - window, 1)
var_om = np.array([omega_dot_n[i:i+window].var() for i in range(0, n_windows, 50)])
best_start = int(np.argmax(var_om) * 50) if len(var_om) else 0
idx_plot = np.arange(best_start, min(best_start + window, N))

fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
fig.suptitle(f'SINDy Model Time-series Fit  (normalized, start={best_start})', fontsize=10)
pred_vx_n = Theta_vx @ c_vx_n
pred_vy_n = Theta_vy @ c_vy_n
pred_om_n = Theta_om @ c_om_n
ts_data = [
    (vx_dot_n[idx_plot],    pred_vx_n[idx_plot],    r'$\dot{v}_{x,n}$ [-]'),
    (vy_dot_n[idx_plot],    pred_vy_n[idx_plot],    r'$\dot{v}_{y,n}$ [-]'),
    (omega_dot_n[idx_plot], pred_om_n[idx_plot],    r'$\dot{\omega}_n$ [-]'),
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
