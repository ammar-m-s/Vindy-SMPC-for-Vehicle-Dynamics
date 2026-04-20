import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. LOAD DATA
# ============================================================
data_path = (Path(__file__).resolve().parent / ".." / "Data" / "sindy_data.npz").resolve()
data = np.load(data_path, allow_pickle=True)

X_sindy    = data['X_sindy']
Xdot_sindy = data['Xdot_sindy']
U_sindy    = data['U_sindy']

# Remove corrupt samples
valid = X_sindy[:, 0] > 1.0
X_sindy    = X_sindy[valid]
Xdot_sindy = Xdot_sindy[valid]
U_sindy    = U_sindy[valid]
N = X_sindy.shape[0]

state_names = ['vx', 'vy', 'omega', 'alpha_f', 'alpha_r']
input_names = ['delta', 'Tr']
dot_names   = ['vx_dot', 'vy_dot', 'omega_dot']

vx    = X_sindy[:, 0]
vy    = X_sindy[:, 1]
omega = X_sindy[:, 2]
af    = X_sindy[:, 3]
ar    = X_sindy[:, 4]
delta = U_sindy[:, 0]
Tr    = U_sindy[:, 1]

print(f"Using {N} samples", flush=True)


# ============================================================
# 2. FULL GENERIC LIBRARY (NO INTENTIONAL DUPLICATES)
#    Theta = [1, X, U, X⊗X, X⊗U, sin(U), sin(X⊗U)]
# ============================================================
def pairwise_products(A, A_names, B, B_names, symmetric=False):
    cols = []
    names = []

    if symmetric and A is B:
        # unique symmetric products only: i <= j
        for i in range(A.shape[1]):
            for j in range(i, A.shape[1]):
                cols.append((A[:, i] * A[:, j]).reshape(-1, 1))
                if i == j:
                    names.append(f"{A_names[i]}^2")
                else:
                    names.append(f"{A_names[i]}*{A_names[j]}")
    else:
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                cols.append((A[:, i] * B[:, j]).reshape(-1, 1))
                names.append(f"{A_names[i]}*{B_names[j]}")

    if len(cols) == 0:
        return np.empty((A.shape[0], 0)), []
    return np.hstack(cols), names


def build_full_generic_library(X, U, x_names, u_names):
    cols = []
    names = []

    # 1
    cols.append(np.ones((X.shape[0], 1)))
    names.append('1')

    # X
    cols.append(X)
    names.extend(x_names)

    # U
    cols.append(U)
    names.extend(u_names)

    # X⊗X (unique symmetric only)
    XX, XX_names = pairwise_products(X, x_names, X, x_names, symmetric=True)
    cols.append(XX)
    names.extend(XX_names)

    # X⊗U
    XU, XU_names = pairwise_products(X, x_names, U, u_names, symmetric=False)
    cols.append(XU)
    names.extend(XU_names)

    # sin(U)
    sinU = np.sin(U)
    sinU_names = [f"sin({name})" for name in u_names]
    cols.append(sinU)
    names.extend(sinU_names)

    # sin(X⊗U)
    sinXU = np.sin(XU)
    sinXU_names = [f"sin({name})" for name in XU_names]
    cols.append(sinXU)
    names.extend(sinXU_names)

    Theta_raw = np.hstack(cols)
    return Theta_raw, names


def dedup_columns(Theta, names, atol=1e-12, rtol=1e-12):
    keep_idx = []
    removed = 0

    for j in range(Theta.shape[1]):
        col_j = Theta[:, j]
        duplicate = False

        for k in keep_idx:
            col_k = Theta[:, k]
            # Remove exact / numerical duplicates only
            if np.allclose(col_j, col_k, atol=atol, rtol=rtol):
                duplicate = True
                break

        if duplicate:
            removed += 1
        else:
            keep_idx.append(j)

    Theta_out = Theta[:, keep_idx]
    names_out = [names[j] for j in keep_idx]
    return Theta_out, names_out, removed


Theta_raw, theta_names_raw = build_full_generic_library(
    X_sindy, U_sindy, state_names, input_names
)
Theta, theta_names, n_removed = dedup_columns(Theta_raw, theta_names_raw)

print(f"Raw full generic library shape: {Theta_raw.shape}", flush=True)
print(f"Deduplicated full generic library shape: {Theta.shape}", flush=True)
print(f"Removed {n_removed} duplicate column(s)", flush=True)

print("\nLibrary terms:")
for i, name in enumerate(theta_names):
    print(f"  [{i:02d}] {name}")


# ============================================================
# 3. STLS
# ============================================================
def stls(Theta, Y, lam, n_iter=10):
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms == 0] = 1.0
    Theta_n = Theta / col_norms

    Phi_n, _, _, _ = np.linalg.lstsq(Theta_n, Y, rcond=None)

    for _ in range(n_iter):
        for j in range(Phi_n.shape[1]):
            active = np.abs(Phi_n[:, j]) >= lam
            if active.sum() == 0:
                Phi_n[:, j] = 0.0
                continue

            Phi_col = np.zeros(Phi_n.shape[0])
            Phi_col[active], _, _, _ = np.linalg.lstsq(Theta_n[:, active], Y[:, j], rcond=None)
            Phi_n[:, j] = Phi_col

    Phi = Phi_n / col_norms[:, None]
    return Phi


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot


def print_sparse_equations(Phi, theta_names, dot_names, zero_tol=1e-12):
    print("\n" + "=" * 90)
    print("IDENTIFIED SPARSE MODEL: Xdot = Theta(X,U) @ Phi")
    print("=" * 90)

    for j, out_name in enumerate(dot_names):
        print(f"\n{out_name} =")
        active_idx = np.where(np.abs(Phi[:, j]) > zero_tol)[0]
        if len(active_idx) == 0:
            print("  0")
            continue
        for idx in active_idx:
            print(f"  {Phi[idx, j]:+.6e} * {theta_names[idx]}")


# ============================================================
# 4. RUN STLS
# ============================================================
# Try values like 5, 10, 15, 20, 30, 50, 100
lambda_val = 15.0
print(f"\nRunning STLS with lambda={lambda_val:.2e} ...", flush=True)

Phi = stls(Theta, Xdot_sindy, lambda_val, n_iter=10)
Xdot_pred = Theta @ Phi

r2s = [r_squared(Xdot_sindy[:, i], Xdot_pred[:, i]) for i in range(3)]
active_counts = [(np.abs(Phi[:, j]) > 1e-12).sum() for j in range(3)]

print(f"\n=== STLS  (lambda={lambda_val:.1e}, N_iter=10) ===")
for name, r2, n_act in zip(dot_names, r2s, active_counts):
    print(f"  R^2({name}) = {r2:.7f}   active terms = {n_act}")

print_sparse_equations(Phi, theta_names, dot_names)


# ============================================================
# 5. SAVE RESULTS
# ============================================================
models_dir = (Path(__file__).resolve().parent / ".." / "Models").resolve()
models_dir.mkdir(parents=True, exist_ok=True)

out_path = models_dir / "sindy_coefficients_full_generic.npz"
np.savez(
    out_path,
    Phi=Phi,
    theta_names=np.array(theta_names, dtype=object),
    Xdot_pred=Xdot_pred,
    R2=np.array(r2s),
    active_counts=np.array(active_counts),
    lambda_val=np.array(lambda_val),
)
print(f"\nSaved {out_path}", flush=True)


# ============================================================
# 6. PLOTS
# ============================================================
plots_dir = (Path(__file__).resolve().parent / ".." / "Plots").resolve()
plots_dir.mkdir(parents=True, exist_ok=True)

def save_current_fig(filename):
    plot_path = plots_dir / filename
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_path}", flush=True)


# --- Parity plot ---
print("Plotting parity figure ...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Full generic SINDy: Predicted vs Actual Derivatives', fontsize=12)

plot_data = [
    (Xdot_sindy[:, 0], Xdot_pred[:, 0], r2s[0], r'$\dot{v}_x$ [m/s$^2$]'),
    (Xdot_sindy[:, 1], Xdot_pred[:, 1], r2s[1], r'$\dot{v}_y$ [m/s$^2$]'),
    (Xdot_sindy[:, 2], Xdot_pred[:, 2], r2s[2], r'$\dot{\omega}$ [rad/s$^2$]'),
]

for ax, (y_true, y_pred, r2, lbl) in zip(axes, plot_data):
    ax.scatter(y_true, y_pred, s=1, alpha=0.2, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.0)
    ax.set_xlabel(f'True {lbl}', fontsize=9)
    ax.set_ylabel(f'SINDy {lbl}', fontsize=9)
    ax.set_title(f'R² = {r2:.6f}', fontsize=10)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
save_current_fig('sindy_plot_parity_full_generic.png')


# --- Coefficients plot ---
print("Plotting coefficient figure ...", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Sparse coefficient matrix Φ for full generic library', fontsize=12)

xs = np.arange(len(theta_names))
for j, ax in enumerate(axes):
    coeffs = Phi[:, j]
    ax.bar(xs, coeffs, linewidth=0.2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel(dot_names[j])
    ax.grid(True, axis='y', alpha=0.3)

step = max(1, len(theta_names) // 20)
axes[-1].set_xticks(xs[::step])
axes[-1].set_xticklabels(np.array(theta_names)[::step], rotation=60, ha='right', fontsize=7)
axes[-1].set_xlabel('Library terms in Θ(X,U)')

fig.tight_layout()
save_current_fig('sindy_plot_coefficients_full_generic.png')


# --- Time-series plot ---
print("Plotting time-series figure ...", flush=True)
window = min(1000, N)
if N > window:
    metric = np.abs(Xdot_sindy[:, 2] - np.mean(Xdot_sindy[:, 2]))
    energy = np.convolve(metric, np.ones(window), mode='valid')
    best_start = int(np.argmax(energy))
else:
    best_start = 0

idx_plot = np.arange(best_start, min(best_start + window, N))

fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
fig.suptitle(f'Full generic SINDy time-series fit (start={best_start})', fontsize=10)

ts_data = [
    (Xdot_sindy[idx_plot, 0], Xdot_pred[idx_plot, 0], r'$\dot{v}_x$ [m/s$^2$]'),
    (Xdot_sindy[idx_plot, 1], Xdot_pred[idx_plot, 1], r'$\dot{v}_y$ [m/s$^2$]'),
    (Xdot_sindy[idx_plot, 2], Xdot_pred[idx_plot, 2], r'$\dot{\omega}$ [rad/s$^2$]'),
]

for ax, (y_true, y_pred, lbl) in zip(axes, ts_data):
    ax.plot(idx_plot, y_true, 'k-', lw=1.0, label='True')
    ax.plot(idx_plot, y_pred, 'r--', lw=0.9, label='SINDy')
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.4)

axes[0].legend(loc='upper right', fontsize=9)
axes[-1].set_xlabel('Sample index')

fig.tight_layout()
save_current_fig('sindy_plot_timeseries_full_generic.png')

print("Done.", flush=True)