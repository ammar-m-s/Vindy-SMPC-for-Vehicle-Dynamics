"""
data_gen_sindy.py  –  Data generation for SINDy vehicle dynamics identification.

Drives a 3-DOF single-track vehicle model (Eq. 9) with Pacejka tire
forces (Eq. 11) around an oval track at multiple reference speeds using
curvature feedforward + pure pursuit (steering) and P+FF speed control.

Outputs:
    sindy_data.npz  – assembled SINDy dataset
    Diagnostic plots: trajectories, velocities, slip angles, tire forces
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from pathlib import Path


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def pacejka(alpha, B, C, D, E):
    Ba = B * alpha
    return D * np.sin(C * np.arctan(Ba - E * (Ba - np.arctan(Ba))))


def get_curvature(s, seg_curvatures, cum_lengths, total_length):
    s = s % total_length
    for i, cl in enumerate(cum_lengths):
        if s < cl:
            return seg_curvatures[i]
    return seg_curvatures[-1]


def vehicle_ode(x, u, p, kappa):
    """3-DOF single-track vehicle model (Equation 9)."""
    s, y, xi, vx, vy, omega, delta, Tr = x
    u1, u2 = u

    # Slip angles (Equation 10) — SAE/bicycle-model convention:
    #   alpha > 0  →  leftward (positive-y) Pacejka force, consistent with the ODE signs.
    #   alpha_f = delta - atan2(vy_front, vx)
    #   alpha_r = -atan2(vy - lr*omega, vx)
    # Equivalently via the wheel-frame velocity rotation followed by negation:
    vy_front = vy + omega * p['lf']
    v_lat_w  = vy_front * np.cos(delta) - vx * np.sin(delta)
    v_lon_w  = vx  * np.cos(delta) + vy_front * np.sin(delta)
    alpha_f  = -np.arctan2(v_lat_w, max(abs(v_lon_w), 0.5))
    alpha_r  = -np.arctan2(vy - omega * p['lr'], max(abs(vx), 0.5))

    # Pacejka lateral forces (Equation 11)
    Ffc = pacejka(alpha_f, p['B_pac'], p['C_pac'], p['D_pac'], p['E_pac'])
    Frc = pacejka(alpha_r, p['B_pac'], p['C_pac'], p['D_pac'], p['E_pac'])

    # Longitudinal drive force and resistances
    Frl  = Tr / p['R_wheel']
    Fres = p['Cr'] * p['M'] * p['g'] + 0.5 * p['rho'] * p['Cd'] * p['Af'] * vx**2

    # Kinematic equations (curvilinear coordinates)
    denom  = max(1.0 - y * kappa, 0.01)
    s_dot  = (vx * np.cos(xi) - vy * np.sin(xi)) / denom
    y_dot  = vx * np.sin(xi) + vy * np.cos(xi)
    xi_dot = omega - kappa * s_dot

    # Dynamic equations (what SINDy will identify)
    vx_dot    = omega * vy + Frl / p['M'] - Ffc * np.sin(delta) / p['M'] - Fres / p['M']
    vy_dot    = -omega * vx + Frc / p['M'] + Ffc * np.cos(delta) / p['M']
    omega_dot = (1.0 / p['Jz']) * (-Frc * p['lr'] + Ffc * p['lf'] * np.cos(delta))

    xdot = np.array([s_dot, y_dot, xi_dot, vx_dot, vy_dot, omega_dot, u1, u2])
    return xdot, alpha_f, alpha_r, Ffc, Frc


def rk4_step(x, u, p, dt, seg_curvatures, cum_lengths, total_length):
    """One RK4 step; returns new state and full derivatives/slip at current x."""
    kap = get_curvature(x[0], seg_curvatures, cum_lengths, total_length)
    k1, af, ar, Ff, Fr = vehicle_ode(x, u, p, kap)

    kap2 = get_curvature(x[0] + dt/2 * k1[0], seg_curvatures, cum_lengths, total_length)
    k2   = vehicle_ode(x + dt/2 * k1, u, p, kap2)[0]

    kap3 = get_curvature(x[0] + dt/2 * k2[0], seg_curvatures, cum_lengths, total_length)
    k3   = vehicle_ode(x + dt/2 * k2, u, p, kap3)[0]

    kap4 = get_curvature(x[0] + dt * k3[0], seg_curvatures, cum_lengths, total_length)
    k4   = vehicle_ode(x + dt  * k3, u, p, kap4)[0]

    x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_new, k1, af, ar, Ff, Fr


# ============================================================
#  1. VEHICLE PARAMETERS (Table 2)
# ============================================================
params = {
    'M':       1412.0,
    'Jz':      1536.7,
    'R_wheel': 0.325,
    'lf':      1.015,
    'lr':      1.895,
    'B_pac':   0.0885 * (180.0 / np.pi),  # convert deg^-1 → rad^-1
    'C_pac':   1.4,
    'D_pac':   8311.0,
    'E_pac':   -2.0,
    'Cr':      0.015,
    'rho':     1.225,
    'Cd':      0.3,
    'Af':      2.2,
    'g':       9.81,
}

# ============================================================
#  2. CONTROLLER GAINS
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
#  3. GENERATE TRACK (oval: 2 straights + 2 semicircles)
# ============================================================
R_turn     = 80.0
L_straight = 150.0

seg_curvatures = np.array([0.0, 1.0/R_turn, 0.0, 1.0/R_turn])
seg_lengths    = np.array([L_straight, np.pi*R_turn, L_straight, np.pi*R_turn])
cum_lengths    = np.cumsum(seg_lengths)
total_length   = cum_lengths[-1]

L_wb = params['lf'] + params['lr']

a_lat_max  = 2.0 * params['D_pac'] / params['M']
v_max_turn = np.sqrt(a_lat_max * R_turn)
print(f"Track: {total_length:.0f} m total  (straights={L_straight:.0f} m, turns R={R_turn:.0f} m)")
print(f"Max feasible turn speed: {v_max_turn:.1f} m/s  (a_lat_max={a_lat_max:.1f} m/s²)\n")

# Precompute centerline for plotting / trajectory reconstruction
ds_track = 0.1
s_vec    = np.arange(0.0, total_length + ds_track, ds_track)
N_pts    = len(s_vec)
X_center = np.zeros(N_pts)
Y_center = np.zeros(N_pts)
theta_c  = np.zeros(N_pts)
theta = 0.0
for i in range(1, N_pts):
    kap       = get_curvature(s_vec[i], seg_curvatures, cum_lengths, total_length)
    theta    += kap * ds_track
    theta_c[i]  = theta
    X_center[i] = X_center[i-1] + np.cos(theta) * ds_track
    Y_center[i] = Y_center[i-1] + np.sin(theta) * ds_track

half_width = 8.0

# ============================================================
#  4. SIMULATION
# ============================================================
dt     = 0.001
v_refs = [10, 15, 20, 25]
n_laps = 2

all_data = []

for v_ref in v_refs:
    print(f"--- v_ref = {v_ref} m/s ---")
    a_req = v_ref**2 / R_turn
    print(f"  Required turn a_lat: {a_req:.1f} m/s²  (max={a_lat_max:.1f})")

    # Initial state: [s, y, xi, vx, vy, omega, delta, Tr]
    Fres_init = (params['Cr'] * params['M'] * params['g']
                 + 0.5 * params['rho'] * params['Cd'] * params['Af'] * v_ref**2)
    Tr_init   = Fres_init * params['R_wheel']
    x = np.array([0.0, 0.0, 0.0, float(v_ref), 0.0, 0.0, 0.0, Tr_init])

    t_max   = n_laps * total_length / max(v_ref, 3.0) * 2.0
    N_steps = int(np.ceil(t_max / dt))

    # Pre-allocate logs
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

        # Lap detection: s crosses a multiple of total_length
        if np.floor(s_cur / total_length) > np.floor(s_prev / total_length):
            lap_count += 1
            print(f"  Lap {lap_count} completed at t = {t:.2f} s")
            if lap_count >= n_laps:
                k_end = k
                break
        s_prev = s_cur

        kap = get_curvature(s_cur, seg_curvatures, cum_lengths, total_length)

        # --- Steering: curvature feedforward + pure pursuit feedback ---
        Ld = ctrl['Ld_min'] + ctrl['Kla'] * max(x[3], 1.0)

        kap_la   = get_curvature(s_cur + Ld, seg_curvatures, cum_lengths, total_length)
        delta_ff = np.arctan(L_wb * kap_la)

        e_la     = x[1] + Ld * np.sin(x[2])
        delta_fb = -np.arctan(2.0 * L_wb * e_la / Ld**2)

        delta_des = np.clip(delta_ff + delta_fb, -ctrl['delta_max'], ctrl['delta_max'])

        # --- Speed: resistance feedforward + proportional feedback ---
        Fres_now = (params['Cr'] * params['M'] * params['g']
                    + 0.5 * params['rho'] * params['Cd'] * params['Af'] * x[3]**2)
        Tr_ff = Fres_now * params['R_wheel']
        e_v   = v_ref - x[3]
        T_des = np.clip(Tr_ff + ctrl['Kp_speed'] * e_v, -ctrl['Tr_max'], ctrl['Tr_max'])

        # --- Rate inputs ---
        u1 = np.clip(ctrl['K_delta'] * (delta_des - x[6]), -ctrl['u1_max'], ctrl['u1_max'])
        u2 = np.clip(ctrl['K_T']     * (T_des     - x[7]), -ctrl['u2_max'], ctrl['u2_max'])
        u  = np.array([u1, u2])

        # RK4 step (logs derivatives at current x)
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
            print(f"  ABORT at t={t:.2f} s: y={x[1]:.1f} m, vx={x[3]:.2f} m/s")
            break

    n = k_end + 1
    all_data.append({
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
    })
    print(f"  {lap_count} laps, {n} samples ({t_log[k_end]:.1f} s)\n")

# ============================================================
#  5. ASSEMBLE SINDy DATASET
# ============================================================
ds_factor  = 10
X_sindy    = []
Xdot_sindy = []
U_sindy    = []

for d in all_data:
    if d['laps'] < 1:
        print(f"WARNING: v_ref={d['v_ref']} did not complete a lap, skipping.")
        continue
    idx = np.arange(0, d['n'], ds_factor)
    X_sindy.append(np.column_stack([
        d['X'][3, idx],      # vx
        d['X'][4, idx],      # vy
        d['X'][5, idx],      # omega
        d['alpha'][0, idx],  # alpha_f
        d['alpha'][1, idx],  # alpha_r
    ]))
    Xdot_sindy.append(np.column_stack([
        d['Xdot'][3, idx],   # vx_dot
        d['Xdot'][4, idx],   # vy_dot
        d['Xdot'][5, idx],   # omega_dot
    ]))
    U_sindy.append(np.column_stack([
        d['X'][6, idx],      # delta
        d['X'][7, idx],      # Tr
    ]))

X_sindy    = np.vstack(X_sindy)
Xdot_sindy = np.vstack(Xdot_sindy)
U_sindy    = np.vstack(U_sindy)

print("=== SINDy Dataset ===")
print(f"  {X_sindy.shape[0]} samples")
print("  X:    [vx, vy, omega, alpha_f, alpha_r]")
print("  Xdot: [vx_dot, vy_dot, omega_dot]")
print("  U:    [delta, Tr]")
out_path = Path(__file__).resolve().parent / "../Data/sindy_data.npz"
out_path.parent.mkdir(parents=True, exist_ok=True)


np.savez(out_path,
         X_sindy=X_sindy, Xdot_sindy=Xdot_sindy, U_sindy=U_sindy)

# ============================================================
#  6. PLOTS
# ============================================================
import sys
_save_only = not sys.stdout.isatty()   # save-only when output is redirected / non-interactive
colors = plt.cm.tab10(np.linspace(0, 0.4, len(v_refs)))

# --- Trajectories ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Reference vs Actual Trajectories")

X_in  = X_center + half_width * np.sin(theta_c)
Y_in  = Y_center - half_width * np.cos(theta_c)
X_out = X_center - half_width * np.sin(theta_c)
Y_out = Y_center + half_width * np.cos(theta_c)

for iv, (d, ax) in enumerate(zip(all_data, axes.flat)):
    ax.plot(X_in,     Y_in,     'k--', lw=1,   label='Boundaries')
    ax.plot(X_out,    Y_out,    'k--', lw=1)
    ax.plot(X_center, Y_center, 'b-',  lw=1.5, label='Reference')

    pidx = np.arange(0, d['n'], 100)
    s_p  = d['X'][0, pidx]
    y_p  = d['X'][1, pidx]
    s_w  = s_p % total_length
    Xc_i = np.interp(s_w, s_vec, X_center)
    Yc_i = np.interp(s_w, s_vec, Y_center)
    tc_i = np.interp(s_w, s_vec, theta_c)
    Xg   = Xc_i - y_p * np.sin(tc_i)
    Yg   = Yc_i + y_p * np.cos(tc_i)
    ax.plot(Xg, Yg, 'r-', lw=1.5, label='Actual')

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f"v_ref = {d['v_ref']} m/s  ({d['laps']} laps)")
    ax.legend(loc='best', fontsize=7)
plt.tight_layout()

# --- Velocities ---
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
fig.suptitle("Velocity Profiles")
ylabels = ['$v_x$ [m/s]', '$v_y$ [m/s]', '$\\omega$ [deg/s]']
rows    = [3, 4, 5]   # 0-indexed state indices
scales  = [1.0, 1.0, 180.0/np.pi]
for j, (ax, label, row, sc) in enumerate(zip(axes, ylabels, rows, scales)):
    for iv, d in enumerate(all_data):
        ax.plot(d['t'], d['X'][row] * sc,
                color=colors[iv], lw=1, label=f"{d['v_ref']} m/s")
    ax.set_ylabel(label); ax.grid(True)
    if j == 0:
        ax.legend(loc='best', fontsize=8)
    if j == 2:
        ax.set_xlabel('Time [s]')
plt.tight_layout()

# --- Slip angles ---
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=False)
fig.suptitle("Slip Angles")
for j, (ax, label) in enumerate(zip(axes, ['$\\alpha_f$ [deg]', '$\\alpha_r$ [deg]'])):
    for iv, d in enumerate(all_data):
        ax.plot(d['t'], d['alpha'][j] * 180/np.pi,
                color=colors[iv], lw=1, label=f"{d['v_ref']} m/s")
    ax.set_ylabel(label); ax.grid(True)
    if j == 0:
        ax.legend(loc='best', fontsize=8)
    else:
        ax.set_xlabel('Time [s]')
plt.tight_layout()

# --- Tire forces vs slip angle ---
a_range = np.linspace(-20, 20, 500) * np.pi / 180
F_ref   = pacejka(a_range, params['B_pac'], params['C_pac'],
                  params['D_pac'], params['E_pac'])
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
fig.suptitle("Tire Forces vs Slip Angle")
for ax, key, flabel, title in zip(
        axes,
        ['Ffc', 'Frc'],
        ['$F_{fc}$ [N]', '$F_{rc}$ [N]'],
        ['Front Lateral Force', 'Rear Lateral Force']):
    ax.plot(a_range*180/np.pi, F_ref, 'k-', lw=2, label='Pacejka')
    for iv, d in enumerate(all_data):
        pidx = np.arange(0, d['n'], 200)
        alpha_idx = 0 if key == 'Ffc' else 1
        ax.scatter(d['alpha'][alpha_idx, pidx]*180/np.pi, d[key][pidx],
                   s=5, color=colors[iv], label=f"{d['v_ref']} m/s")
    ax.set_xlabel('α [deg]'); ax.set_ylabel(flabel)
    ax.set_title(title); ax.legend(loc='best', fontsize=7); ax.grid(True)
plt.tight_layout()

# --- Control inputs ---
fig, axes = plt.subplots(2, 1, figsize=(9, 5))
fig.suptitle("Control Inputs")
for iv, d in enumerate(all_data):
    axes[0].plot(d['t'], d['X'][6]*180/np.pi,
                 color=colors[iv], lw=1, label=f"{d['v_ref']} m/s")
    axes[1].plot(d['t'], d['X'][7], color=colors[iv], lw=1)
axes[0].set_ylabel('δ [deg]'); axes[0].grid(True); axes[0].legend(loc='best', fontsize=8)
axes[1].set_ylabel('$T_r$ [Nm]'); axes[1].set_xlabel('Time [s]'); axes[1].grid(True)
plt.tight_layout()

# --- Slip angle statistics ---
print("--- Slip Angle Coverage ---")
for d in all_data:
    af = d['alpha'][0] * 180/np.pi
    ar = d['alpha'][1] * 180/np.pi
    print(f"  {d['v_ref']:2d} m/s ({d['laps']} laps):  "
          f"alpha_f=[{af.min():+.1f}, {af.max():+.1f}] deg   "
          f"alpha_r=[{ar.min():+.1f}, {ar.max():+.1f}] deg")

for i, fig_label in enumerate(['trajectories', 'velocities', 'slip_angles', 'tire_forces', 'inputs']):
    plt.figure(i + 1)
    out_path2 = Path(__file__).resolve().parent / f'..\Plots\sindy_plot_{fig_label}.png'
    out_path2.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path2, dpi=120, bbox_inches='tight')
    print(f"  Saved to  .\Plots\sindy_plot_{fig_label}.png")

if not _save_only:
    plt.show()
