import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================
ROOT_DIR   = (Path(__file__).resolve().parent / "..").resolve()
DATA_DIR   = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "Models"
PLOTS_DIR  = ROOT_DIR / "Plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "sindy_data.npz"


# ============================================================
# 2. LOAD DATA
# ============================================================
data = np.load(DATA_PATH, allow_pickle=True)
X_sindy    = data['X_sindy']
Xdot_sindy = data['Xdot_sindy']
U_sindy    = data['U_sindy']

N0 = X_sindy.shape[0]
print(f"Loaded {N0} samples from {DATA_PATH}", flush=True)

# Filter out corrupt samples
valid = X_sindy[:, 0] > 1.0
n_removed = int((~valid).sum())
if n_removed:
    print(f"Removed {n_removed} corrupt sample(s) with vx < 1 m/s", flush=True)
    X_sindy    = X_sindy[valid]
    Xdot_sindy = Xdot_sindy[valid]
    U_sindy    = U_sindy[valid]

N = X_sindy.shape[0]
print(f"Using {N} samples", flush=True)

labels_X = ['vx [m/s]', 'vy [m/s]', 'omega [rad/s]', 'alpha_f [rad]', 'alpha_r [rad]']
print("State ranges:", flush=True)
for i, lbl in enumerate(labels_X):
    print(f"  {lbl:<18} [{X_sindy[:, i].min():+.4f}, {X_sindy[:, i].max():+.4f}]", flush=True)

# Unpack columns
vx    = X_sindy[:, 0]
vy    = X_sindy[:, 1]
omega = X_sindy[:, 2]
af    = X_sindy[:, 3]
ar    = X_sindy[:, 4]
delta = U_sindy[:, 0]
Tr    = U_sindy[:, 1]

vx_dot    = Xdot_sindy[:, 0]
vy_dot    = Xdot_sindy[:, 1]
omega_dot = Xdot_sindy[:, 2]


# ============================================================
# 3. NORMAL PAPER LIBRARIES (25-TERM MODEL)
# ============================================================

# vx_dot library (c1 ... c8)
Theta_vx = np.column_stack([
    omega * vy,              # c1
    Tr,                      # c2
    af     * np.sin(delta),  # c3
    af**3  * np.sin(delta),  # c4
    af**5  * np.sin(delta),  # c5
    af**7  * np.sin(delta),  # c6
    np.ones(N),              # c7
    vx**2,                   # c8
])

# vy_dot library (c9 ... c17)
Theta_vy = np.column_stack([
    omega * vx,              # c9
    af     * np.cos(delta),  # c10
    af**3  * np.cos(delta),  # c11
    af**5  * np.cos(delta),  # c12
    af**7  * np.cos(delta),  # c13
    ar,                      # c14
    ar**3,                   # c15
    ar**5,                   # c16
    ar**7,                   # c17
])

# omega_dot library (c18 ... c25)
Theta_om = np.column_stack([
    af     * np.cos(delta),  # c18
    af**3  * np.cos(delta),  # c19
    af**5  * np.cos(delta),  # c20
    af**7  * np.cos(delta),  # c21
    ar,                      # c22
    ar**3,                   # c23
    ar**5,                   # c24
    ar**7,                   # c25
])


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


# ============================================================
# 4. STLS + CONTRIBUTION PRUNING
# ============================================================
def stls_basic(Theta, y, lam_norm, n_iter=10):
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Th = Theta / col_norms

    xi_norm, _, _, _ = np.linalg.lstsq(Th, y, rcond=None)

    for _ in range(n_iter):
        active = np.abs(xi_norm) >= lam_norm
        if active.sum() == 0:
            xi_norm[:] = 0.0
            break
        xi_new = np.zeros_like(xi_norm)
        xi_new[active], _, _, _ = np.linalg.lstsq(Th[:, active], y, rcond=None)
        xi_norm = xi_new

    xi = xi_norm / col_norms
    return xi


def prune_by_contribution(Theta, y, xi, contrib_rel_tol=1e-3):
    """
    Remove terms whose RMS contribution is tiny compared to RMS(y).
    """
    y_rms = np.sqrt(np.mean(y**2))
    if y_rms < 1e-15:
        return xi

    contrib_rms = np.array([
        np.sqrt(np.mean((Theta[:, j] * xi[j])**2))
        for j in range(Theta.shape[1])
    ])

    keep = contrib_rms >= contrib_rel_tol * y_rms
    if keep.sum() == 0:
        return np.zeros_like(xi)

    xi_pruned = np.zeros_like(xi)
    xi_pruned[keep], _, _, _ = np.linalg.lstsq(Theta[:, keep], y, rcond=None)
    return xi_pruned


def stls_with_contribution_pruning(Theta, y, lam_norm=1e-3, contrib_rel_tol=1e-3, n_iter=10):
    """
    First do normal STLS in normalized coordinates, then prune by actual
    contribution in physical coordinates.
    """
    xi = stls_basic(Theta, y, lam_norm=lam_norm, n_iter=n_iter)
    xi = prune_by_contribution(Theta, y, xi, contrib_rel_tol=contrib_rel_tol)
    return xi


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot


def run_all(lam_norm, contrib_rel_tol):
    c_vx = stls_with_contribution_pruning(
        Theta_vx, vx_dot, lam_norm=lam_norm, contrib_rel_tol=contrib_rel_tol, n_iter=10
    )
    c_vy = stls_with_contribution_pruning(
        Theta_vy, vy_dot, lam_norm=lam_norm, contrib_rel_tol=contrib_rel_tol, n_iter=10
    )
    c_om = stls_with_contribution_pruning(
        Theta_om, omega_dot, lam_norm=lam_norm, contrib_rel_tol=contrib_rel_tol, n_iter=10
    )

    pred_vx = Theta_vx @ c_vx
    pred_vy = Theta_vy @ c_vy
    pred_om = Theta_om @ c_om

    r2s = [
        r_squared(vx_dot, pred_vx),
        r_squared(vy_dot, pred_vy),
        r_squared(omega_dot, pred_om),
    ]
    return c_vx, c_vy, c_om, pred_vx, pred_vy, pred_om, r2s


# ============================================================
# 5. RUN IDENTIFICATION
# ============================================================
lambda_val = 10
contrib_rel_tol = 1   # increase to 1e-3 if you want more aggressive pruning

print(f"\nRunning STLS with:", flush=True)
print(f"  lambda_val      = {lambda_val:.1e}", flush=True)
print(f"  contrib_rel_tol = {contrib_rel_tol:.1e}", flush=True)

c_vx, c_vy, c_om, pred_vx, pred_vy, pred_om, r2s = run_all(lambda_val, contrib_rel_tol)

print(f"\n=== STLS  (lambda={lambda_val:.1e}, contribution prune={contrib_rel_tol:.1e}) ===", flush=True)
print(f"  R^2(vx_dot)    = {r2s[0]:.7f}", flush=True)
print(f"  R^2(vy_dot)    = {r2s[1]:.7f}", flush=True)
print(f"  R^2(omega_dot) = {r2s[2]:.7f}", flush=True)

# Optional lambda sweep if fit is poor
if min(r2s) < 0.999:
    print(f"\nmin R^2 = {min(r2s):.6f} < 0.999 -- sweeping lambda ...", flush=True)
    best_score = min(r2s)
    best_res = (c_vx, c_vy, c_om, pred_vx, pred_vy, pred_om, r2s)
    best_lam = lambda_val

    for lam in np.logspace(-6, 0, 80):
        res = run_all(lam, contrib_rel_tol)
        rs = res[-1]
        if min(rs) > best_score:
            best_score = min(rs)
            best_res = res
            best_lam = lam

    lambda_val = best_lam
    c_vx, c_vy, c_om, pred_vx, pred_vy, pred_om, r2s = best_res
    print(f"Best lambda = {lambda_val:.2e}", flush=True)
    print(f"  R^2(vx_dot)    = {r2s[0]:.7f}", flush=True)
    print(f"  R^2(vy_dot)    = {r2s[1]:.7f}", flush=True)
    print(f"  R^2(omega_dot) = {r2s[2]:.7f}", flush=True)

R2_vx, R2_vy, R2_omega = r2s


# ============================================================
# 6. PRINT COEFFICIENTS
# ============================================================
print("\n" + "=" * 60)
print("IDENTIFIED COEFFICIENTS")
print("=" * 60)

for eq_name, coeff_vec, lbls in [
    ('vx_dot',    c_vx, labels['vx']),
    ('vy_dot',    c_vy, labels['vy']),
    ('omega_dot', c_om, labels['om']),
]:
    print(f"\n  {eq_name}:")
    for (cname, term), val in zip(lbls, coeff_vec):
        status = '  ← zeroed' if abs(val) < 1e-15 else ''
        print(f"    {cname:<4}  {term:<18}  {val:+.6e}{status}")
print()


# ============================================================
# 7. PHYSICAL SANITY CHECK
# ============================================================
M       = 1412.0
R_wheel = 0.325
Cr      = 0.015
g       = 9.81
rho     = 1.225
Cd      = 0.3
Af_area = 2.2
Jz      = 1536.7
lf      = 1.015
lr      = 1.895
B_pac   = 0.0885 * (180 / np.pi)
C_pac   = 1.4
D_pac   = 8311.0

Cf_lin = D_pac * C_pac * B_pac

print("=" * 60)
print("PHYSICAL SANITY CHECK")
print("=" * 60)
checks = [
    ('c1',  c_vx[0],  +1.0,                     'Coriolis (expect +1)'),
    ('c2',  c_vx[1],  +1/(M*R_wheel),           'Torque/mass (expect +1/(M*Rw))'),
    ('c7',  c_vx[6],  -Cr*g,                    'Roll. resist. (expect -Cr*g)'),
    ('c8',  c_vx[7],  -0.5*rho*Cd*Af_area/M,    'Aero drag (expect -0.5rhoCdAf/M)'),
    ('c9',  c_vy[0],  -1.0,                     'Coriolis (expect -1)'),
    ('c10', c_vy[1],  +Cf_lin/M,                'Front cornering stiffness/M'),
    ('c14', c_vy[5],  +Cf_lin/M,                'Rear cornering stiffness/M'),
    ('c18', c_om[0],  +Cf_lin*lf/Jz,            'Front moment arm/Jz'),
    ('c22', c_om[4],  -Cf_lin*lr/Jz,            'Rear moment arm/Jz'),
]

print(f"  {'Coeff':<5}  {'Identified':>14}  {'Expected':>14}  {'Ratio':>8}  Description")
print(f"  {'-'*4}   {'-'*13}   {'-'*13}   {'-'*7}  {'-'*30}")
for cname, val, expected, desc in checks:
    ratio = val / expected if abs(expected) > 1e-12 else float('nan')
    print(f"  {cname:<5}  {val:+14.5e}  {expected:+14.5e}  {ratio:8.4f}  {desc}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================
c_all = np.concatenate([c_vx, c_vy, c_om])

out_model = MODELS_DIR / "sindy_coefficients.npz"
np.savez(
    out_model,
    c_vx=c_vx,
    c_vy=c_vy,
    c_omega=c_om,
    c_all=c_all,
    pred_vx=pred_vx,
    pred_vy=pred_vy,
    pred_omega=pred_om,
    R2_vx=np.array(R2_vx),
    R2_vy=np.array(R2_vy),
    R2_omega=np.array(R2_omega),
    lambda_val=np.array(lambda_val),
    contrib_rel_tol=np.array(contrib_rel_tol),
)
print(f"\nSaved {out_model}  (c_all shape: {c_all.shape})", flush=True)


# ============================================================
# 9. PLOTS
# ============================================================
def save_current_fig(filename):
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}", flush=True)


# --- Parity plots ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('SINDy: Predicted vs Actual Derivatives', fontsize=12)
plot_data = [
    (vx_dot,    pred_vx,    R2_vx,    r'$\dot{v}_x$ [m/s$^2$]'),
    (vy_dot,    pred_vy,    R2_vy,    r'$\dot{v}_y$ [m/s$^2$]'),
    (omega_dot, pred_om,    R2_omega, r'$\dot{\omega}$ [rad/s$^2$]'),
]
for ax, (y_true, y_pred, r2, lbl) in zip(axes, plot_data):
    ax.scatter(y_true, y_pred, s=1, alpha=0.2, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y=x')
    ax.set_xlabel(f'True {lbl}', fontsize=9)
    ax.set_ylabel(f'SINDy {lbl}', fontsize=9)
    ax.set_title(f'R^2 = {r2:.7f}', fontsize=10)
    ax.grid(True, alpha=0.4)
fig.tight_layout()
save_current_fig('sindy_plot_parity.png')

# --- Coefficient bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Identified SINDy Coefficients (grey = zeroed)', fontsize=11)
for ax, (coeff_vec, lbls, title) in zip(axes, [
    (c_vx, labels['vx'], r'$\dot{v}_x$ (c1-c8)'),
    (c_vy, labels['vy'], r'$\dot{v}_y$ (c9-c17)'),
    (c_om, labels['om'], r'$\dot{\omega}$ (c18-c25)'),
]):
    xlbls = [f"{n}\n{t}" for n, t in lbls]
    bar_clr = ['steelblue' if abs(v) > 1e-15 else 'lightgrey' for v in coeff_vec]
    ax.bar(xlbls, coeff_vec, color=bar_clr, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.grid(True, axis='y', alpha=0.4)
fig.tight_layout()
save_current_fig('sindy_plot_coefficients.png')

# --- Time-series plot ---
window = min(1000, N)
if N > window:
    var_om = np.array([omega_dot[i:i+window].var() for i in range(0, N - window, 50)])
    best_start = int(np.argmax(var_om) * 50)
else:
    best_start = 0

idx_plot = np.arange(best_start, min(best_start + window, N))

fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
fig.suptitle(f'SINDy Model Time-series Fit (start={best_start})', fontsize=10)
ts_data = [
    (vx_dot[idx_plot],    pred_vx[idx_plot],    r'$\dot{v}_x$ [m/s$^2$]'),
    (vy_dot[idx_plot],    pred_vy[idx_plot],    r'$\dot{v}_y$ [m/s$^2$]'),
    (omega_dot[idx_plot], pred_om[idx_plot],    r'$\dot{\omega}$ [rad/s$^2$]'),
]
for ax, (y_true, y_pred, lbl) in zip(axes, ts_data):
    ax.plot(idx_plot, y_true, 'k-', lw=1.0, label='True')
    ax.plot(idx_plot, y_pred, 'r--', lw=0.8, label='SINDy')
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.4)
    if ax is axes[0]:
        ax.legend(loc='upper right', fontsize=9)
axes[-1].set_xlabel('Sample index')
fig.tight_layout()
save_current_fig('sindy_plot_timeseries.png')

print("Done.", flush=True)