"""
sindy_mpc.py  –  SINDy-identified vehicle model + Model Predictive Control
                  tested on the same oval track used for data generation.

Pipeline:
  1. Re-identify SINDy model from sindy_data.npz (if available), else use
     hard-coded physics-derived coefficients.
  2. Build a linearised MPC around each operating point using the SINDy model
     as the prediction model.
  3. Drive the full nonlinear Pacejka simulator around the oval and compare
     against the baseline pure-pursuit + P controller.
  4. Save results and plots.

State (SINDy): z = [vx, vy, omega]   (3-DOF)
Inputs:        u = [delta, Tr]        (steering angle, rear wheel torque)
MPC optimises delta and Tr increments over a receding horizon.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import block_diag
from scipy.linalg import expm


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "Models"
PLOTS_DIR  = ROOT_DIR / "Plots"
for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# VEHICLE PARAMETERS  (identical to data_gen_sindy.py)
# ─────────────────────────────────────────────────────────────
p = {
    'M':       1412.0,
    'Jz':      1536.7,
    'R_wheel': 0.325,
    'lf':      1.015,
    'lr':      1.895,
    'B_pac':   0.0885 * (180.0 / np.pi),
    'C_pac':   1.4,
    'D_pac':   8311.0,
    'E_pac':   -2.0,
    'Cr':      0.015,
    'rho':     1.225,
    'Cd':      0.3,
    'Af':      2.2,
    'g':       9.81,
}
L_wb = p['lf'] + p['lr']

# # ─────────────────────────────────────────────────────────────
# # TRACK  (oval: 2 straights + 2 semicircles)
# # ─────────────────────────────────────────────────────────────
R_turn     = 80.0
L_straight = 150.0
seg_curvatures = np.array([0.0, 1.0/R_turn, 0.0, 1.0/R_turn])
seg_lengths    = np.array([L_straight, np.pi*R_turn, L_straight, np.pi*R_turn])
cum_lengths    = np.cumsum(seg_lengths)
total_length   = cum_lengths[-1]

def get_curvature(s):
    s_mod = s % total_length
    for i, cl in enumerate(cum_lengths):
        if s_mod < cl:
            return seg_curvatures[i]
    return seg_curvatures[-1]

# Centerline for plotting
ds_track = 0.2
s_vec    = np.arange(0.0, total_length + ds_track, ds_track)
X_center = np.zeros(len(s_vec))
Y_center = np.zeros(len(s_vec))
theta_c  = np.zeros(len(s_vec))
theta = 0.0
for i in range(1, len(s_vec)):
    kap          = get_curvature(s_vec[i])
    theta       += kap * ds_track
    theta_c[i]   = theta
    X_center[i]  = X_center[i-1] + np.cos(theta) * ds_track
    Y_center[i]  = Y_center[i-1] + np.sin(theta) * ds_track

half_width = 8.0


# ─────────────────────────────────────────────────────────────
# PACEJKA + VEHICLE ODE  (truth model / simulator)
# ─────────────────────────────────────────────────────────────
def pacejka(alpha):
    Ba = p['B_pac'] * alpha
    return p['D_pac'] * np.sin(p['C_pac'] * np.arctan(
        Ba - p['E_pac'] * (Ba - np.arctan(Ba))))

def vehicle_ode(x, u):
    """Full 8-state ODE; returns xdot and slip angles."""
    s, y, xi, vx, vy, omega, delta, Tr = x
    u1, u2 = u
    kappa  = get_curvature(s)

    vy_front = vy + omega * p['lf']
    v_lat_w  = vy_front * np.cos(delta) - vx * np.sin(delta)
    v_lon_w  = vx * np.cos(delta) + vy_front * np.sin(delta)
    alpha_f  = -np.arctan2(v_lat_w, max(abs(v_lon_w), 0.5))
    alpha_r  = -np.arctan2(vy - omega * p['lr'], max(abs(vx), 0.5))

    Ffc  = pacejka(alpha_f)
    Frc  = pacejka(alpha_r)
    Frl  = Tr / p['R_wheel']
    Fres = p['Cr'] * p['M'] * p['g'] + 0.5 * p['rho'] * p['Cd'] * p['Af'] * vx**2

    denom  = max(1.0 - y * kappa, 0.01)
    s_dot  = (vx * np.cos(xi) - vy * np.sin(xi)) / denom
    y_dot  = vx * np.sin(xi) + vy * np.cos(xi)
    xi_dot = omega - kappa * s_dot

    vx_dot    = omega * vy + Frl / p['M'] - Ffc * np.sin(delta) / p['M'] - Fres / p['M']
    vy_dot    = -omega * vx + Frc / p['M'] + Ffc * np.cos(delta) / p['M']
    omega_dot = (Ffc * p['lf'] * np.cos(delta) - Frc * p['lr']) / p['Jz']

    return np.array([s_dot, y_dot, xi_dot, vx_dot, vy_dot, omega_dot, u1, u2]), alpha_f, alpha_r

def rk4_step(x, u, dt):
    k1, af, ar = vehicle_ode(x, u)
    k2 = vehicle_ode(x + dt/2 * k1, u)[0]
    k3 = vehicle_ode(x + dt/2 * k2, u)[0]
    k4 = vehicle_ode(x + dt  * k3, u)[0]
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4), af, ar

# ─────────────────────────────────────────────────────────────
# SINDY MODEL  (from sindy_identification.py library)
# ─────────────────────────────────────────────────────────────
def try_load_sindy_coefficients():
    """Try loading pre-identified coefficients; return None on failure."""
    cpath = MODELS_DIR / "sindy_coefficients.npz"
    if not cpath.exists():
        return None
    try:
        d = np.load(cpath, allow_pickle=True)
        return d['c_vx'], d['c_vy'], d['c_omega']
    except Exception:
        return None

def sindy_identify_from_data():
    """Run STLS on sindy_data.npz; returns (c_vx, c_vy, c_om) or None."""
    dpath = DATA_DIR / "sindy_data.npz"
    if not dpath.exists():
        return None
    print("  Identifying SINDy model from data ...", flush=True)
    data = np.load(dpath, allow_pickle=True)
    X    = data['X_sindy']
    Xd   = data['Xdot_sindy']
    U    = data['U_sindy']
    valid = X[:, 0] > 1.0
    X, Xd, U = X[valid], Xd[valid], U[valid]
    N = X.shape[0]
    vx, vy, omega, af, ar = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    delta, Tr = U[:,0], U[:,1]
    vx_dot, vy_dot, omega_dot = Xd[:,0], Xd[:,1], Xd[:,2]

    Theta_vx = np.column_stack([
        omega * vy, Tr,
        af*np.sin(delta), af**3*np.sin(delta), af**5*np.sin(delta), af**7*np.sin(delta),
        np.ones(N), vx**2,
    ])
    Theta_vy = np.column_stack([
        omega * vx,
        af*np.cos(delta), af**3*np.cos(delta), af**5*np.cos(delta), af**7*np.cos(delta),
        ar, ar**3, ar**5, ar**7,
    ])
    Theta_om = np.column_stack([
        af*np.cos(delta), af**3*np.cos(delta), af**5*np.cos(delta), af**7*np.cos(delta),
        ar, ar**3, ar**5, ar**7,
    ])

    def stls(Th, y, lam=0.001, n_iter=10):
        norms = np.linalg.norm(Th, axis=0); norms[norms==0] = 1.0
        Tn = Th / norms
        xi, _, _, _ = np.linalg.lstsq(Tn, y, rcond=None)
        for _ in range(n_iter):
            act = np.abs(xi) >= lam
            if not act.any(): xi[:] = 0.0; break
            xn = np.zeros_like(xi)
            xn[act], _, _, _ = np.linalg.lstsq(Tn[:, act], y, rcond=None)
            xi = xn
        return xi / norms

    c_vx = stls(Theta_vx, vx_dot)
    c_vy = stls(Theta_vy, vy_dot)
    c_om = stls(Theta_om, omega_dot)
    return c_vx, c_vy, c_om

def get_sindy_model():
    res = try_load_sindy_coefficients()
    if res is not None:
        print("  Loaded SINDy coefficients from Models/sindy_coefficients.npz", flush=True)
        return res
    res = sindy_identify_from_data()
    if res is not None:
        return res
    # Fallback: physics-based coefficients
    print("  WARNING: No data found — using physics-based fallback coefficients.", flush=True)
    Cf = p['D_pac'] * p['C_pac'] * p['B_pac']
    # c_vx: [omega*vy, Tr, af*sin, af^3*sin, af^5*sin, af^7*sin, 1, vx^2]
    c_vx = np.array([1.0, 1/(p['M']*p['R_wheel']), -Cf/p['M'], 0, 0, 0,
                     -p['Cr']*p['g'], -0.5*p['rho']*p['Cd']*p['Af']/p['M']])
    c_vy = np.array([-1.0, Cf/p['M'], 0, 0, 0, Cf/p['M'], 0, 0, 0])
    c_om = np.array([Cf*p['lf']/p['Jz'], 0, 0, 0, -Cf*p['lr']/p['Jz'], 0, 0, 0])
    return c_vx, c_vy, c_om

print("Loading SINDy model ...", flush=True)
c_vx, c_vy, c_om = get_sindy_model()
print(f"  c_vx shape: {c_vx.shape}, c_vy shape: {c_vy.shape}, c_om shape: {c_om.shape}", flush=True)

# ─────────────────────────────────────────────────────────────
# SINDY PREDICTION  (one RK4 step using identified model)
# ─────────────────────────────────────────────────────────────
def sindy_predict(vx, vy, omega, delta, Tr, dt, n_sub=5):
    """Integrate SINDy dynamics with small sub-steps."""
    dt_sub = dt / n_sub
    for _ in range(n_sub):
        vy_front = vy + omega * p['lf']
        v_lat_w  = vy_front * np.cos(delta) - vx * np.sin(delta)
        v_lon_w  = vx * np.cos(delta) + vy_front * np.sin(delta)
        af = -np.arctan2(v_lat_w, max(abs(v_lon_w), 0.5))
        ar = -np.arctan2(vy - omega * p['lr'], max(abs(vx), 0.5))

        # SINDy features
        Phi_vx = np.array([omega*vy, Tr,
                            af*np.sin(delta), af**3*np.sin(delta),
                            af**5*np.sin(delta), af**7*np.sin(delta),
                            1.0, vx**2])
        Phi_vy = np.array([omega*vx,
                            af*np.cos(delta), af**3*np.cos(delta),
                            af**5*np.cos(delta), af**7*np.cos(delta),
                            ar, ar**3, ar**5, ar**7])
        Phi_om = np.array([af*np.cos(delta), af**3*np.cos(delta),
                            af**5*np.cos(delta), af**7*np.cos(delta),
                            ar, ar**3, ar**5, ar**7])

        vx_dot    = float(c_vx @ Phi_vx)
        vy_dot    = float(c_vy @ Phi_vy)
        omega_dot = float(c_om @ Phi_om)

        vx    += dt_sub * vx_dot
        vy    += dt_sub * vy_dot
        omega += dt_sub * omega_dot
        vx = max(vx, 0.5)
    return vx, vy, omega

# ─────────────────────────────────────────────────────────────
# LINEARISED SINDY JACOBIANS  (for MPC)
# ─────────────────────────────────────────────────────────────
def sindy_jacobians(vx0, vy0, om0, delta0, Tr0):
    """
    Compute A = df/dz,  B = df/du  at operating point (z0, u0).
    z = [vx, vy, omega],  u = [delta, Tr]
    Returns continuous-time A (3x3), B (3x2).
    """
    eps_s = 1e-5
    eps_u = 1e-4

    def f(vx, vy, om, delta, Tr):
        vy_f   = vy + om * p['lf']
        v_lat  = vy_f * np.cos(delta) - vx * np.sin(delta)
        v_lon  = vx * np.cos(delta) + vy_f * np.sin(delta)
        af     = -np.arctan2(v_lat, max(abs(v_lon), 0.5))
        ar     = -np.arctan2(vy - om * p['lr'], max(abs(vx), 0.5))
        Ph_vx = np.array([om*vy, Tr,
                           af*np.sin(delta), af**3*np.sin(delta),
                           af**5*np.sin(delta), af**7*np.sin(delta),
                           1.0, vx**2])
        Ph_vy = np.array([om*vx,
                           af*np.cos(delta), af**3*np.cos(delta),
                           af**5*np.cos(delta), af**7*np.cos(delta),
                           ar, ar**3, ar**5, ar**7])
        Ph_om = np.array([af*np.cos(delta), af**3*np.cos(delta),
                           af**5*np.cos(delta), af**7*np.cos(delta),
                           ar, ar**3, ar**5, ar**7])
        return np.array([c_vx @ Ph_vx, c_vy @ Ph_vy, c_om @ Ph_om])

    f0 = f(vx0, vy0, om0, delta0, Tr0)
    A  = np.zeros((3, 3))
    for i, (dvx, dvy, dom) in enumerate([
            (eps_s, 0, 0), (0, eps_s, 0), (0, 0, eps_s)]):
        A[:, i] = (f(vx0+dvx, vy0+dvy, om0+dom, delta0, Tr0) - f0) / eps_s

    B = np.zeros((3, 2))
    for i, (ddel, dTr) in enumerate([(eps_u, 0), (0, eps_u)]):
        B[:, i] = (f(vx0, vy0, om0, delta0+ddel, Tr0+dTr) - f0) / eps_u

    return A, B

# ─────────────────────────────────────────────────────────────
# MPC SETUP
# ─────────────────────────────────────────────────────────────
Np   = 20
dt = 0.02

nz = 5   # [vx, vy, omega, y, xi]
nu = 2   # [delta, Tr]

Q_mpc = np.diag([2000.0, 30.0, 30.0, 500.0, 500.0])
P_mpc = Q_mpc * 2.0
R_mpc = np.diag([1.0, 0.0001])

u_min = np.array([-0.5, -2000])
u_max = np.array([0.5, 2000])


du_min = np.array([-2.0 * dt, -15000.0 * dt])
du_max = np.array([ 2.0 * dt,  15000.0 * dt])

def discretize_AB(A, B, dt):
    M = np.block([
        [A, B],
        [np.zeros((B.shape[1], A.shape[0] + B.shape[1]))]
    ])
    Md = expm(M * dt)
    Ad = Md[:A.shape[0], :A.shape[1]]
    Bd = Md[:A.shape[0], A.shape[1]:]
    return Ad, Bd

def mpc_solve(Ad, Bd, z_err):
    Fx = np.zeros((nz * Np, nz))
    Fu = np.zeros((nz * Np, nu * Np))

    Ak = np.eye(nz)
    for k in range(Np):
        Ak = Ad @ Ak
        Fx[k*nz:(k+1)*nz] = Ak
        for j in range(k+1):
            Aj = np.linalg.matrix_power(Ad, k-j)
            Fu[k*nz:(k+1)*nz, j*nu:(j+1)*nu] = Aj @ Bd

    Qbar = block_diag(*[Q_mpc]*(Np-1) + [P_mpc])
    Rbar = block_diag(*[R_mpc]*Np)

    H = Fu.T @ Qbar @ Fu + Rbar
    f = Fu.T @ Qbar @ Fx @ z_err

    try:
        U = -np.linalg.solve(H + 1e-6*np.eye(H.shape[0]), f)
    except np.linalg.LinAlgError:
        U = np.zeros(nu*Np)

    u = U[:nu]
    u = np.clip(u, u_min, u_max)

    return u

# ─────────────────────────────────────────────────────────────
# BASELINE CONTROLLER  (same as data generation)
# ─────────────────────────────────────────────────────────────
ctrl_base = {
    'Ld_min':    8.0,
    'Kla':       0.6,
    'Kp_speed':  3000.0,
    'K_delta':   10.0,
    'K_T':       15.0,
    'delta_max': 30,
    'Tr_max':    5000,
    'u1_max':    2.0,
    'u2_max':    15000.0,
}
def build_augmented_model(vx, vy, omega, y, xi, delta, Tr, s,
                          sindy_jacobians, get_curvature):

    A_dyn, B_dyn = sindy_jacobians(vx, vy, omega, delta, Tr)

    A = np.zeros((5,5))
    B = np.zeros((5,2))

    # Dynamics
    A[:3,:3] = A_dyn
    B[:3,:]  = B_dyn

    kappa = get_curvature(s)

    # Kinematics
    A[3,0] = np.sin(xi)
    A[3,1] = np.cos(xi)
    A[3,4] = vx*np.cos(xi) - vy*np.sin(xi)

    A[4,2] = 1.0
    A[4,0] = -kappa   

    return A, B
def baseline_control(x, v_ref):
    s, y, xi, vx, vy, omega, delta, Tr = x
    Ld       = ctrl_base['Ld_min'] + ctrl_base['Kla'] * max(vx, 1.0)
    kap_la   = get_curvature(s + Ld)
    delta_ff = np.arctan(L_wb * kap_la)
    e_la     = y + Ld * np.sin(xi)
    delta_fb = -np.arctan(2.0 * L_wb * e_la / Ld**2)
    delta_des= np.clip(delta_ff + delta_fb, -ctrl_base['delta_max'], ctrl_base['delta_max'])

    Fres_now = p['Cr']*p['M']*p['g'] + 0.5*p['rho']*p['Cd']*p['Af']*vx**2
    Tr_ff    = Fres_now * p['R_wheel']
    T_des    = np.clip(Tr_ff + ctrl_base['Kp_speed'] * (v_ref - vx),
                       -ctrl_base['Tr_max'], ctrl_base['Tr_max'])

    u1 = np.clip(ctrl_base['K_delta'] * (delta_des - delta), -ctrl_base['u1_max'], ctrl_base['u1_max'])
    u2 = np.clip(ctrl_base['K_T']     * (T_des     - Tr),    -ctrl_base['u2_max'], ctrl_base['u2_max'])
    return np.array([u1, u2])

# ─────────────────────────────────────────────────────────────
# SINDY-MPC CONTROLLER
# ─────────────────────────────────────────────────────────────


# ------------------------------------------------------------
# MAIN MPC CONTROLLER
# ------------------------------------------------------------
def sindy_mpc_control(x, v_ref, u_prev, dt_mpc_ctrl,
                      ctrl_base, sindy_jacobians, get_curvature):

    s, y, xi, vx, vy, omega, delta, Tr = x

    kappa = get_curvature(s)

    z = np.array([vx, vy, omega, y, xi])
    z_ref = np.array([v_ref, 0.0, kappa*v_ref, 0.0, 0.0])
    z_err = z - z_ref

    A, B = build_augmented_model(
        vx, vy, omega, y, xi, delta, Tr, s,
        sindy_jacobians, get_curvature
    )

    Ad, Bd = discretize_AB(A, B, dt_mpc_ctrl)

    u_abs = mpc_solve(Ad, Bd, z_err)

    delta_des, Tr_des = u_abs

    # Convert to rate inputs
    u1 = np.clip(
        ctrl_base['K_delta'] * (delta_des - delta),
        -ctrl_base['u1_max'], ctrl_base['u1_max']
    )

    u2 = np.clip(
        ctrl_base['K_T'] * (Tr_des - Tr),
        -ctrl_base['u2_max'], ctrl_base['u2_max']
    )

    return np.array([u1, u2]), u_abs

# ─────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────
def run_simulation(v_ref, controller='mpc', n_laps=2, dt_sim=0.001, dt_mpc_ctrl=0.02):
    print(f"\n{'='*55}", flush=True)
    print(f"  {controller.upper()}  |  v_ref={v_ref} m/s  |  {n_laps} laps", flush=True)
    print(f"{'='*55}", flush=True)

    Fres_init = p['Cr']*p['M']*p['g'] + 0.5*p['rho']*p['Cd']*p['Af']*v_ref**2
    Tr_init   = Fres_init * p['R_wheel']
    x = np.array([0.0, 0.0, 0.0, float(v_ref), 0.0, 0.0, 0.0, Tr_init])

    t_max   = n_laps * total_length / max(v_ref, 3.0) * 2.5
    N_steps = int(np.ceil(t_max / dt_sim))

    X_log   = np.zeros((8, N_steps))
    U_log   = np.zeros((2, N_steps))
    t_log   = np.zeros(N_steps)
    af_log  = np.zeros(N_steps)
    ar_log  = np.zeros(N_steps)

    u_prev     = np.array([0.0, Tr_init])
    mpc_step   = max(1, int(dt_mpc_ctrl / dt_sim))

    u = np.array([0.0, 0.0])  # rate inputs
    lap_count = 0
    s_prev    = 0.0
    k_end     = 0

    for k in range(N_steps):
        t     = k * dt_sim
        s_cur = x[0]

        if np.floor(s_cur / total_length) > np.floor(s_prev / total_length):
            lap_count += 1
            print(f"  Lap {lap_count} at t={t:.2f} s  vx={x[3]:.1f} m/s  y={x[1]:.2f} m",
                  flush=True)
            if lap_count >= n_laps:
                k_end = k
                break
        s_prev = s_cur

        # Update control at MPC rate
        if k % mpc_step == 0:
            if controller == 'mpc':
                u, u_prev = sindy_mpc_control(
                    x, v_ref, u_prev, dt_mpc_ctrl,
                    ctrl_base, sindy_jacobians, get_curvature
                )
            else:
                u = baseline_control(x, v_ref)
                u_prev = np.array([x[6], x[7]])

        X_log[:, k] = x
        U_log[:, k] = u
        t_log[k]    = t

        x_new, af, ar = rk4_step(x, u, dt_sim)
        af_log[k] = af
        ar_log[k] = ar

        k_end = k
        x = x_new

        if abs(x[1]) > 50 or x[3] < 0.5:
            print(f"  ABORT at t={t:.2f}s: y={x[1]:.1f}m  vx={x[3]:.2f}m/s", flush=True)
            break

    n = k_end + 1
    print(f"  Done: {lap_count} laps, {n} steps, t={t_log[k_end]:.2f} s", flush=True)
    return {
        'controller': controller,
        'v_ref':      v_ref,
        'X':          X_log[:, :n],
        'U':          U_log[:, :n],
        't':          t_log[:n],
        'af':         af_log[:n],
        'ar':         ar_log[:n],
        'laps':       lap_count,
        'n':          n,
    }

# ─────────────────────────────────────────────────────────────
# RUN BOTH CONTROLLERS AT TWO SPEEDS
# ─────────────────────────────────────────────────────────────
v_refs_test = [15, 20]
results = []
for v_ref in v_refs_test:
    res_base = run_simulation(v_ref, controller='baseline', n_laps=2)
    res_mpc  = run_simulation(v_ref, controller='mpc',      n_laps=2)
    results.append((res_base, res_mpc))

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Controller':<12} {'v_ref':>6}  {'Laps':>5}  {'RMS y_err':>10}  {'Max |y|':>9}  {'RMS vx_err':>11}")
print("="*65)
for res_base, res_mpc in results:
    for res in [res_base, res_mpc]:
        y_err  = res['X'][1]
        vx_err = res['X'][3] - res['v_ref']
        print(f"  {res['controller']:<10} {res['v_ref']:>6}  {res['laps']:>5}  "
              f"{np.sqrt(np.mean(y_err**2)):>10.4f}  "
              f"{np.abs(y_err).max():>9.4f}  "
              f"{np.sqrt(np.mean(vx_err**2)):>11.4f}")
print()

# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def world_coords(d):
    """Convert curvilinear (s, y) to Cartesian world coordinates."""
    s_arr = d['X'][0]
    y_arr = d['X'][1]
    step  = max(1, len(s_arr) // 3000)
    idx   = np.arange(0, len(s_arr), step)
    s_w   = s_arr[idx] % total_length
    yp    = y_arr[idx]
    Xc    = np.interp(s_w, s_vec, X_center)
    Yc    = np.interp(s_w, s_vec, Y_center)
    tc    = np.interp(s_w, s_vec, theta_c)
    Xg    = Xc - yp * np.sin(tc)
    Yg    = Yc + yp * np.cos(tc)
    return Xg, Yg

colors_ctrl = {'baseline': '#e74c3c', 'mpc': '#2ecc71'}
labels_ctrl = {'baseline': 'Baseline PP+P', 'mpc': 'SINDy-MPC'}

# Boundaries
X_in  = X_center + half_width * np.sin(theta_c)
Y_in  = Y_center - half_width * np.cos(theta_c)
X_out = X_center - half_width * np.sin(theta_c)
Y_out = Y_center + half_width * np.cos(theta_c)

# ── Figure 1: Trajectories ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Track Trajectories: Baseline vs SINDy-MPC', fontsize=14, fontweight='bold')
for ax, (res_base, res_mpc), v_ref in zip(axes, results, v_refs_test):
    ax.fill(np.concatenate([X_out, X_in[::-1]]),
            np.concatenate([Y_out, Y_in[::-1]]),
            color='#e8e8e8', zorder=0)
    ax.plot(X_center, Y_center, 'k--', lw=1.5, label='Centreline', zorder=1)
    for res in [res_base, res_mpc]:
        Xg, Yg = world_coords(res)
        c = colors_ctrl[res['controller']]
        ax.plot(Xg, Yg, color=c, lw=1.8, label=labels_ctrl[res['controller']], zorder=2)
    ax.set_aspect('equal')
    ax.set_title(f'v_ref = {v_ref} m/s', fontsize=12)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'sindy_mpc_trajectories.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved {PLOTS_DIR / 'sindy_mpc_trajectories.png'}", flush=True)

# ── Figure 2: Lateral deviation and velocity ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex='col')
fig.suptitle('Lateral Deviation and Longitudinal Velocity', fontsize=13, fontweight='bold')
for col_idx, (res_base, res_mpc) in enumerate(results):
    ax_y  = axes[0, col_idx]
    ax_vx = axes[1, col_idx]
    for res in [res_base, res_mpc]:
        t = res['t']
        c = colors_ctrl[res['controller']]
        lbl = labels_ctrl[res['controller']]
        ax_y.plot(t,  res['X'][1],     color=c, lw=1.3, label=lbl)
        ax_vx.plot(t, res['X'][3],     color=c, lw=1.3, label=lbl)
    ax_y.axhline(0,            color='k', lw=0.8, ls='--')
    ax_y.axhline( half_width,  color='grey', lw=0.6, ls=':')
    ax_y.axhline(-half_width,  color='grey', lw=0.6, ls=':')
    ax_vx.axhline(res_base['v_ref'], color='k', lw=0.8, ls='--', label='v_ref')
    ax_y.set_ylabel('Lateral deviation y [m]')
    ax_vx.set_ylabel('vx [m/s]')
    ax_vx.set_xlabel('Time [s]')
    ax_y.set_title(f'v_ref = {res_base["v_ref"]} m/s')
    ax_y.legend(fontsize=8); ax_y.grid(True, alpha=0.3)
    ax_vx.legend(fontsize=8); ax_vx.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'sindy_mpc_states.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved {PLOTS_DIR / 'sindy_mpc_states.png'}", flush=True)

# ── Figure 3: Steering and torque inputs ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex='col')
fig.suptitle('Control Inputs: Baseline vs SINDy-MPC', fontsize=13, fontweight='bold')
for col_idx, (res_base, res_mpc) in enumerate(results):
    ax_d = axes[0, col_idx]
    ax_T = axes[1, col_idx]
    for res in [res_base, res_mpc]:
        t   = res['t']
        c   = colors_ctrl[res['controller']]
        lbl = labels_ctrl[res['controller']]
        ax_d.plot(t, res['X'][6] * 180/np.pi, color=c, lw=1.2, label=lbl)
        ax_T.plot(t, res['X'][7],              color=c, lw=1.2, label=lbl)
    ax_d.set_ylabel('δ [deg]')
    ax_T.set_ylabel('Tr [Nm]')
    ax_T.set_xlabel('Time [s]')
    ax_d.set_title(f'v_ref = {res_base["v_ref"]} m/s')
    ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)
    ax_T.legend(fontsize=8); ax_T.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'sindy_mpc_inputs.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved {PLOTS_DIR / 'sindy_mpc_inputs.png'}", flush=True)

# ── Figure 4: Slip angles ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex='col')
fig.suptitle('Tire Slip Angles: Baseline vs SINDy-MPC', fontsize=13, fontweight='bold')
for col_idx, (res_base, res_mpc) in enumerate(results):
    ax_af = axes[0, col_idx]
    ax_ar = axes[1, col_idx]
    for res in [res_base, res_mpc]:
        t   = res['t']
        c   = colors_ctrl[res['controller']]
        lbl = labels_ctrl[res['controller']]
        ax_af.plot(t, res['af'] * 180/np.pi, color=c, lw=1.1, label=lbl)
        ax_ar.plot(t, res['ar'] * 180/np.pi, color=c, lw=1.1, label=lbl)
    ax_af.set_ylabel('α_f [deg]')
    ax_ar.set_ylabel('α_r [deg]')
    ax_ar.set_xlabel('Time [s]')
    ax_af.set_title(f'v_ref = {res_base["v_ref"]} m/s')
    ax_af.legend(fontsize=8); ax_af.grid(True, alpha=0.3)
    ax_ar.legend(fontsize=8); ax_ar.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'sindy_mpc_slip.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved {PLOTS_DIR / 'sindy_mpc_slip.png'}", flush=True)

# ── Figure 5: R² scatter of SINDy model vs Pacejka truth ─────
print("\nEvaluating SINDy model prediction accuracy ...", flush=True)
dpath = DATA_DIR / "sindy_data.npz"
if dpath.exists():
    raw = np.load(dpath, allow_pickle=True)
    Xs = raw['X_sindy']; Xds = raw['Xdot_sindy']; Us = raw['U_sindy']
    valid = Xs[:, 0] > 1.0
    Xs, Xds, Us = Xs[valid], Xds[valid], Us[valid]
    N = Xs.shape[0]
    vx0, vy0, om0 = Xs[:,0], Xs[:,1], Xs[:,2]
    af0, ar0      = Xs[:,3], Xs[:,4]
    delta0, Tr0   = Us[:,0], Us[:,1]

    Ph_vx = np.column_stack([om0*vy0, Tr0,
                              af0*np.sin(delta0), af0**3*np.sin(delta0),
                              af0**5*np.sin(delta0), af0**7*np.sin(delta0),
                              np.ones(N), vx0**2])
    Ph_vy = np.column_stack([om0*vx0,
                              af0*np.cos(delta0), af0**3*np.cos(delta0),
                              af0**5*np.cos(delta0), af0**7*np.cos(delta0),
                              ar0, ar0**3, ar0**5, ar0**7])
    Ph_om = np.column_stack([af0*np.cos(delta0), af0**3*np.cos(delta0),
                              af0**5*np.cos(delta0), af0**7*np.cos(delta0),
                              ar0, ar0**3, ar0**5, ar0**7])

    p_vx = Ph_vx @ c_vx
    p_vy = Ph_vy @ c_vy
    p_om = Ph_om @ c_om

    def r2(yt, yp):
        ss_res = np.sum((yt-yp)**2)
        ss_tot = np.sum((yt-yt.mean())**2)
        return 1 - ss_res/ss_tot if ss_tot > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('SINDy Model Parity (used inside MPC predictor)', fontsize=12, fontweight='bold')
    for ax, yt, yp, lbl in zip(axes,
                                [Xds[:,0], Xds[:,1], Xds[:,2]],
                                [p_vx, p_vy, p_om],
                                [r'$\dot{v}_x$', r'$\dot{v}_y$', r'$\dot{\omega}$']):
        r2v = r2(yt, yp)
        ax.scatter(yt, yp, s=1, alpha=0.15, color='steelblue', rasterized=True)
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
        ax.set_xlabel(f'True {lbl}'); ax.set_ylabel(f'SINDy {lbl}')
        ax.set_title(f'R² = {r2v:.6f}'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'sindy_mpc_parity.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved {PLOTS_DIR / 'sindy_mpc_parity.png'}", flush=True)

print("\nAll done.", flush=True)