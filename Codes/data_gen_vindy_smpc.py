"""
data_gen_vindy.py  –  Data generation for VINDy vehicle dynamics identification.

Drives a 3-DOF single-track vehicle model with Pacejka tire forces around an
oval track at multiple reference speeds.  Key differences from SINDy data gen:
  • Independent front/rear Pacejka parameters
  • Gaussian uncertainty on Pacejka (B,C,D,E), mass M, and CG position lf
  • Fres DROPPED entirely (set to zero)
  • Multiple trajectories with different parameter samples
  • Downsampling factor = 30  (dt_eff = 0.03 s)
  • Explicit train / validation split
  • Saves parameter samples alongside trajectory data for ground truth checks

Outputs:
    vindy_data.npz  –  training + validation dataset with parameter records
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def pacejka(alpha, B, C, D, E):
    """Pacejka magic formula lateral tire force."""
    Ba = B * alpha
    return D * np.sin(C * np.arctan(Ba - E * (Ba - np.arctan(Ba))))


def get_curvature(s, seg_curvatures, cum_lengths, total_length):
    s = s % total_length
    for i, cl in enumerate(cum_lengths):
        if s < cl:
            return seg_curvatures[i]
    return seg_curvatures[-1]


def vehicle_ode(x, u, p, kappa):
    """
    3-DOF single-track vehicle model.
    Modified: independent front/rear Pacejka, Fres = 0.
    """
    s, y, xi, vx, vy, omega, delta, Tr = x
    u1, u2 = u

    # Slip angles — SAE/bicycle-model convention
    vy_front = vy + omega * p['lf']
    v_lat_w  = vy_front * np.cos(delta) - vx * np.sin(delta)
    v_lon_w  = vx * np.cos(delta) + vy_front * np.sin(delta)
    alpha_f  = -np.arctan2(v_lat_w, max(abs(v_lon_w), 0.5))
    alpha_r  = -np.arctan2(vy - omega * p['lr'], max(abs(vx), 0.5))

    # Pacejka lateral forces — INDEPENDENT front/rear parameters
    Ffc = pacejka(alpha_f, p['B_f'], p['C_f'], p['D_f'], p['E_f'])
    Frc = pacejka(alpha_r, p['B_r'], p['C_r'], p['D_r'], p['E_r'])

    # Longitudinal drive force
    Frl = Tr / p['R_wheel']

    # Fres = 0  (dropped by design decision)

    # Kinematic equations (curvilinear coordinates)
    denom  = max(1.0 - y * kappa, 0.01)
    s_dot  = (vx * np.cos(xi) - vy * np.sin(xi)) / denom
    y_dot  = vx * np.sin(xi) + vy * np.cos(xi)
    xi_dot = omega - kappa * s_dot

    # Dynamic equations (what VINDy will identify)
    vx_dot    = omega * vy + Frl / p['M'] - Ffc * np.sin(delta) / p['M']
    vy_dot    = -omega * vx + Frc / p['M'] + Ffc * np.cos(delta) / p['M']
    omega_dot = (1.0 / p['Jz']) * (-Frc * p['lr'] + Ffc * p['lf'] * np.cos(delta))

    xdot = np.array([s_dot, y_dot, xi_dot, vx_dot, vy_dot, omega_dot, u1, u2])
    return xdot, alpha_f, alpha_r, Ffc, Frc


def rk4_step(x, u, p, dt, seg_curvatures, cum_lengths, total_length):
    """One RK4 step; returns new state and exact RHS derivatives at current x."""
    kap = get_curvature(x[0], seg_curvatures, cum_lengths, total_length)
    k1, af, ar, Ff, Fr = vehicle_ode(x, u, p, kap)

    kap2 = get_curvature(x[0] + dt/2 * k1[0], seg_curvatures, cum_lengths, total_length)
    k2   = vehicle_ode(x + dt/2 * k1, u, p, kap2)[0]

    kap3 = get_curvature(x[0] + dt/2 * k2[0], seg_curvatures, cum_lengths, total_length)
    k3   = vehicle_ode(x + dt/2 * k2, u, p, kap3)[0]

    kap4 = get_curvature(x[0] + dt * k3[0], seg_curvatures, cum_lengths, total_length)
    k4   = vehicle_ode(x + dt * k3, u, p, kap4)[0]

    x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_new, k1, af, ar, Ff, Fr


# ============================================================
#  1. NOMINAL VEHICLE PARAMETERS
# ============================================================
nominal = {
    'M':       1412.0,
    'Jz':      1536.7,
    'R_wheel': 0.325,
    'lf':      1.015,
    'lr':      1.895,
    # Front Pacejka (nominal)
    'B_f':     0.0885 * (180.0 / np.pi),   # convert deg^-1 → rad^-1
    'C_f':     1.4,
    'D_f':     8311.0,
    'E_f':     -2.0,
    # Rear Pacejka (nominal — same as front at baseline)
    'B_r':     0.0885 * (180.0 / np.pi),
    'C_r':     1.4,
    'D_r':     8311.0,
    'E_r':     -2.0,
    # Unused but kept for compatibility
    'g':       9.81,
}

L_wb = nominal['lf'] + nominal['lr']   # wheelbase — fixed

# ============================================================
#  2. GAUSSIAN UNCERTAINTY SPECIFICATION
# ============================================================
uncertainty = {
    # Pacejka front
    'B_f':  0.05 * nominal['B_f'],        # 5%
    'C_f':  0.03 * nominal['C_f'],        # 3%
    'D_f':  0.10 * nominal['D_f'],        # 10%
    'E_f':  0.05 * abs(nominal['E_f']),   # 5% of |E|
    # Pacejka rear (independent)
    'B_r':  0.05 * nominal['B_r'],
    'C_r':  0.03 * nominal['C_r'],
    'D_r':  0.10 * nominal['D_r'],
    'E_r':  0.05 * abs(nominal['E_r']),
    # Mass and CG
    'M':    0.08 * nominal['M'],          # 8%
    'lf':   0.04 * nominal['lf'],         # 4%
}

# ============================================================
#  3. CONTROLLER GAINS (unchanged from SINDy)
# ============================================================
ctrl = {
    'Ld_min':    8.0,
    'Kla':       0.6,
    'Kp_speed':  3000.0,
    'K_delta':   10.0,
    'K_T':       15.0,
    'delta_max': 30.0 * np.pi / 180.0,
    'Tr_max':    5000.0,
    'u1_max':    2.0,
    'u2_max':    15000.0,
}

# ============================================================
#  4. TRACK GEOMETRY (oval: 2 straights + 2 semicircles)
# ============================================================
R_turn     = 80.0
L_straight = 150.0

seg_curvatures = np.array([0.0, 1.0/R_turn, 0.0, 1.0/R_turn])
seg_lengths    = np.array([L_straight, np.pi*R_turn, L_straight, np.pi*R_turn])
cum_lengths    = np.cumsum(seg_lengths)
total_length   = cum_lengths[-1]

print(f"Track: {total_length:.0f} m total  "
      f"(straights={L_straight:.0f} m, turns R={R_turn:.0f} m)")

# ============================================================
#  5. PARAMETER SAMPLING FUNCTION
# ============================================================
def sample_parameters(nominal, uncertainty, rng=None):
    """
    Sample a parameter dictionary from Gaussian distributions.
    Applies physical constraints after sampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = dict(nominal)    # copy nominal values

    # Sample uncertain parameters
    for key in uncertainty:
        p[key] = rng.normal(nominal[key], uncertainty[key])

    # --- Physical constraints ---
    # E must be ≤ 1.0
    p['E_f'] = min(p['E_f'], 1.0)
    p['E_r'] = min(p['E_r'], 1.0)

    # Mass must stay reasonable
    p['M'] = max(p['M'], 0.7 * nominal['M'])

    # CG must stay between axles
    p['lf'] = np.clip(p['lf'], 0.3 * L_wb, 0.7 * L_wb)
    p['lr'] = L_wb - p['lf']

    # Jz scales roughly with M (approximation: Jz ∝ M)
    p['Jz'] = nominal['Jz'] * (p['M'] / nominal['M'])

    return p


# ============================================================
#  6. SINGLE TRAJECTORY SIMULATION
# ============================================================
def simulate_trajectory(params, v_ref, n_laps, dt, ctrl,
                        seg_curvatures, cum_lengths, total_length):
    """
    Simulate one trajectory with given parameters and reference speed.
    Returns logged data dictionary.
    """
    L_wb_local = params['lf'] + params['lr']

    # Initial state: [s, y, xi, vx, vy, omega, delta, Tr]
    # No Fres → initial Tr = 0 for steady state at constant speed
    # (in practice controller will find equilibrium quickly)
    x = np.array([0.0, 0.0, 0.0, float(v_ref), 0.0, 0.0, 0.0, 0.0])

    t_max   = n_laps * total_length / max(v_ref, 3.0) * 1.5
    N_steps = int(np.ceil(t_max / dt))

    # Pre-allocate
    X_log     = np.zeros((8, N_steps))
    Xdot_log  = np.zeros((8, N_steps))
    U_log     = np.zeros((2, N_steps))
    t_log     = np.zeros(N_steps)
    alpha_log = np.zeros((2, N_steps))
    Ffc_log   = np.zeros(N_steps)
    Frc_log   = np.zeros(N_steps)

    k_end     = 0
    lap_count = 0
    s_prev    = 0.0

    for k in range(N_steps):
        t     = k * dt
        s_cur = x[0]

        # Lap detection
        if np.floor(s_cur / total_length) > np.floor(s_prev / total_length):
            lap_count += 1
            if lap_count >= n_laps:
                k_end = k
                break
        s_prev = s_cur

        kap = get_curvature(s_cur, seg_curvatures, cum_lengths, total_length)

        # --- Steering: curvature feedforward + pure pursuit ---
        Ld = ctrl['Ld_min'] + ctrl['Kla'] * max(x[3], 1.0)
        kap_la   = get_curvature(s_cur + Ld, seg_curvatures, cum_lengths, total_length)
        delta_ff = np.arctan(L_wb_local * kap_la)
        e_la     = x[1] + Ld * np.sin(x[2])
        delta_fb = -np.arctan(2.0 * L_wb_local * e_la / Ld**2)
        delta_des = np.clip(delta_ff + delta_fb, -ctrl['delta_max'], ctrl['delta_max'])

        # --- Speed: proportional feedback only (no Fres feedforward) ---
        e_v   = v_ref - x[3]
        T_des = np.clip(ctrl['Kp_speed'] * e_v, -ctrl['Tr_max'], ctrl['Tr_max'])

        # --- Rate inputs ---
        u1 = np.clip(ctrl['K_delta'] * (delta_des - x[6]), -ctrl['u1_max'], ctrl['u1_max'])
        u2 = np.clip(ctrl['K_T']     * (T_des     - x[7]), -ctrl['u2_max'], ctrl['u2_max'])
        u  = np.array([u1, u2])

        # RK4 step — k1 = exact RHS at current state
        x_new, xdot, af, ar, Ff, Fr = rk4_step(
            x, u, params, dt, seg_curvatures, cum_lengths, total_length)

        k_end            = k
        X_log[:, k]      = x
        Xdot_log[:, k]   = xdot
        U_log[:, k]      = u
        t_log[k]         = t
        alpha_log[:, k]  = [af, ar]
        Ffc_log[k]       = Ff
        Frc_log[k]       = Fr

        x = x_new

        # Safety abort
        if abs(x[1]) > 50 or x[3] < 0.1:
            break

    n = k_end + 1
    return {
        'v_ref':  v_ref,
        'X':      X_log[:, :n],
        'Xdot':   Xdot_log[:, :n],
        'U':      U_log[:, :n],
        't':      t_log[:n],
        'alpha':  alpha_log[:, :n],
        'Ffc':    Ffc_log[:n],
        'Frc':    Frc_log[:n],
        'n':      n,
        'laps':   lap_count,
    }


# ============================================================
#  7. MAIN GENERATION LOOP
# ============================================================
dt       = 0.005
v_refs   = [15,20,25]
n_laps   = 1
ds_factor = 6       # downsample: dt_eff = 0.03 s (same effective rate)

N_traj_train = 200   # training trajectories (each gets all v_refs)
N_traj_val   = 3    # validation trajectories

rng = np.random.default_rng(42)

def generate_dataset(N_traj, label, rng):
    """Generate N_traj trajectories, each running all v_refs."""
    X_all    = []
    Xd_all   = []
    U_all    = []
    param_records = []

    for i in range(N_traj):
        # Sample parameters for this trajectory
        if label == 'val' and i == 0:
            # First validation trajectory = nominal (for clean comparison)
            p_i = dict(nominal)
        else:
            p_i = sample_parameters(nominal, uncertainty, rng)

        param_record = {k: p_i[k] for k in
                        ['B_f','C_f','D_f','E_f','B_r','C_r','D_r','E_r',
                         'M','Jz','lf','lr']}
        param_records.append(param_record)

        for v_ref in v_refs:
            d = simulate_trajectory(
                p_i, v_ref, n_laps, dt, ctrl,
                seg_curvatures, cum_lengths, total_length)

            if d['laps'] < 1:
                print(f"  WARNING: traj {i}, v_ref={v_ref} did not complete a lap, skipping.")
                continue

            idx = np.arange(0, d['n'], ds_factor)
            X_all.append(np.column_stack([
                d['X'][3, idx],      # vx
                d['X'][4, idx],      # vy
                d['X'][5, idx],      # omega
                d['alpha'][0, idx],  # alpha_f
                d['alpha'][1, idx],  # alpha_r
            ]))
            Xd_all.append(np.column_stack([
                d['Xdot'][3, idx],   # vx_dot
                d['Xdot'][4, idx],   # vy_dot
                d['Xdot'][5, idx],   # omega_dot
            ]))
            U_all.append(np.column_stack([
                d['X'][6, idx],      # delta
                d['X'][7, idx],      # Tr
            ]))

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{label}] Completed trajectory {i+1}/{N_traj}")

    X_all  = np.vstack(X_all)
    Xd_all = np.vstack(Xd_all)
    U_all  = np.vstack(U_all)

    # Filter: remove samples where vx < 1.0 m/s
    valid = X_all[:, 0] > 1.0
    X_all  = X_all[valid]
    Xd_all = Xd_all[valid]
    U_all  = U_all[valid]

    return X_all, Xd_all, U_all, param_records


print("\n=== Generating TRAINING data ===")
X_train, Xd_train, U_train, params_train = generate_dataset(N_traj_train, 'train', rng)

print(f"\n=== Generating VALIDATION data ===")
X_val, Xd_val, U_val, params_val = generate_dataset(N_traj_val, 'val', rng)


# ============================================================
#  8. SAVE
# ============================================================
out_dir = Path(__file__).resolve().parent / "Data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "vindy_data.npz"

# Convert param records to arrays for saving
def params_to_arrays(records):
    keys = list(records[0].keys())
    return {k: np.array([r[k] for r in records]) for k in keys}

params_train_arr = params_to_arrays(params_train)
params_val_arr   = params_to_arrays(params_val)

np.savez(out_path,
         # Training data
         X_train=X_train, Xd_train=Xd_train, U_train=U_train,
         # Validation data
         X_val=X_val, Xd_val=Xd_val, U_val=U_val,
         # Parameter records (for ground truth comparison)
         **{f'ptrain_{k}': v for k, v in params_train_arr.items()},
         **{f'pval_{k}':   v for k, v in params_val_arr.items()},
         # Nominal parameters
         **{f'nom_{k}': np.array(nominal[k]) for k in
            ['B_f','C_f','D_f','E_f','B_r','C_r','D_r','E_r',
             'M','Jz','lf','lr','R_wheel']},
         )

print(f"\n=== VINDy Dataset Saved to {out_path} ===")
print(f"  Training:   {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  X:    [vx, vy, omega, alpha_f, alpha_r]")
print(f"  Xdot: [vx_dot, vy_dot, omega_dot]")
print(f"  U:    [delta, Tr]")
print(f"  Downsample factor: {ds_factor}  (dt_eff = {dt*ds_factor:.3f} s)")
print(f"  N_traj_train = {N_traj_train},  N_traj_val = {N_traj_val}")


# ============================================================
#  9. DIAGNOSTIC PLOTS
# ============================================================
plot_dir = Path(__file__).resolve().parent / "Plots"
plot_dir.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    path = plot_dir / name
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# --- 9a. Parameter distributions ---
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Sampled Parameter Distributions (Training)', fontsize=14)

param_keys = ['B_f','C_f','D_f','E_f', 'B_r','C_r','D_r','E_r', 'M','Jz','lf','lr']
for ax, key in zip(axes.flat, param_keys):
    vals = params_train_arr[key]
    ax.hist(vals, bins=20, edgecolor='k', alpha=0.7)
    ax.axvline(nominal.get(key, vals.mean()), color='r', ls='--', lw=2, label='nominal')
    ax.set_title(key)
    ax.legend(fontsize=7)
fig.tight_layout()
save_fig(fig, 'vindy_param_distributions.png')


# --- 9b. Slip angle coverage ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Slip Angle Coverage (Training Data)')
axes[0].hist(X_train[:, 3] * 180/np.pi, bins=100, edgecolor='k', alpha=0.7)
axes[0].set_xlabel('α_f [deg]')
axes[0].set_title('Front slip angle')
axes[0].grid(True, alpha=0.3)
axes[1].hist(X_train[:, 4] * 180/np.pi, bins=100, edgecolor='k', alpha=0.7)
axes[1].set_xlabel('α_r [deg]')
axes[1].set_title('Rear slip angle')
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, 'vindy_slip_angle_coverage.png')


# --- 9c. Tire force curves: true Pacejka for all sampled params ---
a_range = np.linspace(-15, 15, 500) * np.pi / 180

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('True Pacejka Curves — All Training Parameter Samples')

for i in range(len(params_train)):
    p = params_train[i]
    F_f = pacejka(a_range, p['B_f'], p['C_f'], p['D_f'], p['E_f'])
    F_r = pacejka(a_range, p['B_r'], p['C_r'], p['D_r'], p['E_r'])
    axes[0].plot(a_range*180/np.pi, F_f, 'b-', alpha=0.15, lw=0.5)
    axes[1].plot(a_range*180/np.pi, F_r, 'r-', alpha=0.15, lw=0.5)

# Nominal curve
F_nom = pacejka(a_range, nominal['B_f'], nominal['C_f'],
                nominal['D_f'], nominal['E_f'])
axes[0].plot(a_range*180/np.pi, F_nom, 'k-', lw=2, label='Nominal')
axes[1].plot(a_range*180/np.pi, F_nom, 'k-', lw=2, label='Nominal')

for ax, title in zip(axes, ['Front Tire', 'Rear Tire']):
    ax.set_xlabel('α [deg]')
    ax.set_ylabel('F_y [N]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
save_fig(fig, 'vindy_true_pacejka_curves.png')


# --- 9d. Derivative distributions ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Derivative Distributions (Training)')
dot_names = [r'$\dot{v}_x$', r'$\dot{v}_y$', r'$\dot{\omega}$']
for ax, col, name in zip(axes, range(3), dot_names):
    ax.hist(Xd_train[:, col], bins=100, edgecolor='k', alpha=0.7)
    ax.set_xlabel(name)
    ax.set_title(f'{name} distribution')
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, 'vindy_derivative_distributions.png')


# --- 9e. State statistics ---
print("\n--- Training Data Statistics ---")
state_names = ['vx', 'vy', 'omega', 'alpha_f', 'alpha_r']
for i, name in enumerate(state_names):
    vals = X_train[:, i]
    scale = 180/np.pi if 'alpha' in name else 1.0
    unit  = 'deg' if 'alpha' in name else ('[m/s]' if 'v' in name else '[rad/s]')
    print(f"  {name:8s}: min={vals.min()*scale:+8.3f}, max={vals.max()*scale:+8.3f}, "
          f"mean={vals.mean()*scale:+8.3f}, std={vals.std()*scale:8.3f} {unit}")

print("\n--- Derivative Statistics ---")
dot_labels = ['vx_dot', 'vy_dot', 'omega_dot']
for i, name in enumerate(dot_labels):
    vals = Xd_train[:, i]
    print(f"  {name:10s}: min={vals.min():+10.4f}, max={vals.max():+10.4f}, "
          f"mean={vals.mean():+10.4f}, std={vals.std():10.4f}")

print("\nDone.")
