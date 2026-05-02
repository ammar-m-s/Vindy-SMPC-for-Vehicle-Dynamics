"""
data_gen_vindy_smpc_rich_excitation.py

Richer data generation for VINDy vehicle dynamics identification.

Main changes relative to the original oval-track closed-loop generator:
  • Keeps the same 3-DOF single-track vehicle model with independent front/rear Pacejka tires.
  • Keeps the same identification arrays expected by the notebook:
        X_*  = [vx, vy, omega, alpha_f, alpha_r]
        Xd_* = [vx_dot, vy_dot, omega_dot]
        U_*  = [delta, Tr]
  • Adds richer excitation so the library columns are less collinear:
        - time-varying speed reference
        - steering multisine + chirp + PRBS injection
        - torque multisine + PRBS injection
        - randomized initial lateral/yaw/velocity offsets
        - randomized oval geometry per parameter trajectory
        - more speed bases and an untouched test split
  • Removes the strict "must complete a lap" requirement.
  • Uses fixed-duration excited runs plus minimum-duration and physical-validity filters.
  • Randomizes the starting path coordinate s0 so short runs still cover straights and turns.
  • Saves trajectory/time/track metadata so you can evaluate by trajectory, not only by stacked samples.
  • Caps each accepted run using a contiguous fixed-dt window, preserving time consistency.

Outputs:
    Data/vindy_data_rich_excitation.npz

Use this file when the original data produces high condition number / collinearity in Theta.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def make_oval_track(R_turn=80.0, L_straight=150.0):
    """Return curvature and length arrays for an oval track."""
    seg_curvatures = np.array([0.0, 1.0 / R_turn, 0.0, 1.0 / R_turn])
    seg_lengths = np.array([L_straight, np.pi * R_turn, L_straight, np.pi * R_turn])
    cum_lengths = np.cumsum(seg_lengths)
    total_length = cum_lengths[-1]
    return seg_curvatures, seg_lengths, cum_lengths, total_length


def sample_track_geometry(rng, label):
    """
    Randomize the track geometry to avoid learning only one curvature pattern.

    The ranges are intentionally moderate. Since rich-excitation runs are duration-based,
    completion of a full lap is not required. Validation/test use the same broad
    family but different random draws.
    """
    if label == "train":
        R_turn = rng.uniform(55.0, 120.0)
        L_straight = rng.uniform(90.0, 220.0)
    elif label == "val":
        R_turn = rng.uniform(60.0, 130.0)
        L_straight = rng.uniform(100.0, 240.0)
    else:  # test
        R_turn = rng.uniform(50.0, 135.0)
        L_straight = rng.uniform(80.0, 260.0)
    return R_turn, L_straight, *make_oval_track(R_turn, L_straight)


def get_curvature(s, seg_curvatures, cum_lengths, total_length):
    s = s % total_length
    for i, cl in enumerate(cum_lengths):
        if s < cl:
            return seg_curvatures[i]
    return seg_curvatures[-1]


def vehicle_ode(x, u, p, kappa):
    """
    3-DOF single-track vehicle model.

    State x = [s, y, xi, vx, vy, omega, delta, Tr]
    Input u = [delta_dot_cmd, Tr_dot_cmd]
    """
    s, y, xi, vx, vy, omega, delta, Tr = x
    u1, u2 = u

    # Slip angles — SAE/bicycle-model convention
    vy_front = vy + omega * p["lf"]
    v_lat_w = vy_front * np.cos(delta) - vx * np.sin(delta)
    v_lon_w = vx * np.cos(delta) + vy_front * np.sin(delta)
    alpha_f = -np.arctan2(v_lat_w, max(abs(v_lon_w), 0.5))
    alpha_r = -np.arctan2(vy - omega * p["lr"], max(abs(vx), 0.5))

    # Pacejka lateral forces — independent front/rear parameters
    Ffc = pacejka(alpha_f, p["B_f"], p["C_f"], p["D_f"], p["E_f"])
    Frc = pacejka(alpha_r, p["B_r"], p["C_r"], p["D_r"], p["E_r"])

    # Longitudinal drive force
    Frl = Tr / p["R_wheel"]

    # Kinematic equations in curvilinear coordinates
    denom = max(1.0 - y * kappa, 0.01)
    s_dot = (vx * np.cos(xi) - vy * np.sin(xi)) / denom
    y_dot = vx * np.sin(xi) + vy * np.cos(xi)
    xi_dot = omega - kappa * s_dot

    # Dynamic equations identified by VINDy
    vx_dot = omega * vy + Frl / p["M"] - Ffc * np.sin(delta) / p["M"]
    vy_dot = -omega * vx + Frc / p["M"] + Ffc * np.cos(delta) / p["M"]
    omega_dot = (1.0 / p["Jz"]) * (-Frc * p["lr"] + Ffc * p["lf"] * np.cos(delta))

    xdot = np.array([s_dot, y_dot, xi_dot, vx_dot, vy_dot, omega_dot, u1, u2])
    return xdot, alpha_f, alpha_r, Ffc, Frc


def rk4_step(x, u, p, dt, seg_curvatures, cum_lengths, total_length):
    """One RK4 step; returns new state and exact RHS derivatives at current x."""
    kap = get_curvature(x[0], seg_curvatures, cum_lengths, total_length)
    k1, af, ar, Ff, Fr = vehicle_ode(x, u, p, kap)

    kap2 = get_curvature(x[0] + dt / 2 * k1[0], seg_curvatures, cum_lengths, total_length)
    k2 = vehicle_ode(x + dt / 2 * k1, u, p, kap2)[0]

    kap3 = get_curvature(x[0] + dt / 2 * k2[0], seg_curvatures, cum_lengths, total_length)
    k3 = vehicle_ode(x + dt / 2 * k2, u, p, kap3)[0]

    kap4 = get_curvature(x[0] + dt * k3[0], seg_curvatures, cum_lengths, total_length)
    k4 = vehicle_ode(x + dt * k3, u, p, kap4)[0]

    x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_new, k1, af, ar, Ff, Fr


def piecewise_constant_signal(N_steps, dt, amp, switch_time, rng):
    """Random telegraph / PRBS-like signal with random block amplitudes."""
    block_len = max(1, int(round(switch_time / dt)))
    n_blocks = int(np.ceil(N_steps / block_len)) + 1
    signs = rng.choice([-1.0, 1.0], size=n_blocks)
    mags = rng.uniform(0.35, 1.0, size=n_blocks)
    blocks = amp * signs * mags
    return np.repeat(blocks, block_len)[:N_steps]


def build_excitation_signals(N_steps, dt, v_base, rng, strength="rich"):
    """
    Build bounded excitation signals used by the closed-loop controller.

    The excitation is injected at the desired steering angle and desired wheel torque,
    not directly into the derivative commands. This keeps actuator dynamics active,
    so the identified inputs delta and Tr remain realistic.
    """
    t = np.arange(N_steps) * dt

    if strength == "none":
        return {
            "v_ref_profile": np.full(N_steps, v_base),
            "delta_add": np.zeros(N_steps),
            "T_add": np.zeros(N_steps),
        }

    # --- Time-varying speed reference ---
    # Slow variations make Tr not just a nearly deterministic function of speed error.
    v_amp_1 = rng.uniform(0.6, 1.8)        # m/s
    v_amp_2 = rng.uniform(0.2, 0.8)        # m/s
    f_v1 = rng.uniform(0.015, 0.045)       # Hz
    f_v2 = rng.uniform(0.050, 0.110)       # Hz
    ph_v1 = rng.uniform(0, 2 * np.pi)
    ph_v2 = rng.uniform(0, 2 * np.pi)
    v_ref_profile = (
        v_base
        + v_amp_1 * np.sin(2 * np.pi * f_v1 * t + ph_v1)
        + v_amp_2 * np.sin(2 * np.pi * f_v2 * t + ph_v2)
    )
    v_ref_profile = np.clip(v_ref_profile, 8.0, 32.0)

    # --- Steering excitation: multisine + chirp + PRBS ---
    # Keep the magnitude moderate to avoid making many trajectories abort.
    deg = np.pi / 180.0
    delta_ms = (
        rng.uniform(0.6, 1.4) * deg * np.sin(2 * np.pi * rng.uniform(0.05, 0.12) * t + rng.uniform(0, 2 * np.pi))
        + rng.uniform(0.3, 0.9) * deg * np.sin(2 * np.pi * rng.uniform(0.14, 0.30) * t + rng.uniform(0, 2 * np.pi))
    )
    T_total = max(t[-1], dt)
    f0 = rng.uniform(0.02, 0.05)
    f1 = rng.uniform(0.25, 0.55)
    chirp_phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T_total * t**2)
    delta_chirp = rng.uniform(0.3, 0.8) * deg * np.sin(chirp_phase + rng.uniform(0, 2 * np.pi))
    delta_prbs = piecewise_constant_signal(
        N_steps, dt, amp=rng.uniform(0.4, 1.2) * deg,
        switch_time=rng.uniform(0.35, 0.90), rng=rng
    )
    delta_add = delta_ms + delta_chirp + delta_prbs

    # Bound injected steering excitation to avoid unrealistic maneuvers.
    delta_add = np.clip(delta_add, -4.0 * deg, 4.0 * deg)

    # --- Torque excitation: lower-frequency multisine + PRBS pulses ---
    T_ms = (
        rng.uniform(180.0, 550.0) * np.sin(2 * np.pi * rng.uniform(0.025, 0.080) * t + rng.uniform(0, 2 * np.pi))
        + rng.uniform(100.0, 350.0) * np.sin(2 * np.pi * rng.uniform(0.090, 0.180) * t + rng.uniform(0, 2 * np.pi))
    )
    T_prbs = piecewise_constant_signal(
        N_steps, dt, amp=rng.uniform(120.0, 500.0),
        switch_time=rng.uniform(0.60, 1.50), rng=rng
    )
    T_add = np.clip(T_ms + T_prbs, -900.0, 900.0)

    return {
        "v_ref_profile": v_ref_profile,
        "delta_add": delta_add,
        "T_add": T_add,
    }


# ============================================================
#  1. NOMINAL VEHICLE PARAMETERS
# ============================================================
nominal = {
    "M": 1412.0,
    "Jz": 1536.7,
    "R_wheel": 0.325,
    "lf": 1.015,
    "lr": 1.895,
    "B_f": 0.0885 * (180.0 / np.pi),
    "C_f": 1.4,
    "D_f": 8311.0,
    "E_f": -2.0,
    "B_r": 0.0885 * (180.0 / np.pi),
    "C_r": 1.4,
    "D_r": 8311.0,
    "E_r": -2.0,
    "g": 9.81,
}

L_wb = nominal["lf"] + nominal["lr"]

# ============================================================
#  2. GAUSSIAN UNCERTAINTY SPECIFICATION
# ============================================================
uncertainty = {
    "B_f": 0.05 * nominal["B_f"],
    "C_f": 0.03 * nominal["C_f"],
    "D_f": 0.10 * nominal["D_f"],
    "E_f": 0.05 * abs(nominal["E_f"]),
    "B_r": 0.05 * nominal["B_r"],
    "C_r": 0.03 * nominal["C_r"],
    "D_r": 0.10 * nominal["D_r"],
    "E_r": 0.05 * abs(nominal["E_r"]),
    "M": 0.08 * nominal["M"],
    "lf": 0.04 * nominal["lf"],
}

# ============================================================
#  3. CONTROLLER GAINS
# ============================================================
ctrl = {
    "Ld_min": 8.0,
    "Kla": 0.6,
    "Kp_speed": 3000.0,
    "K_delta": 10.0,
    "K_T": 15.0,
    "delta_max": 30.0 * np.pi / 180.0,
    "Tr_max": 5000.0,
    "u1_max": 2.0,
    "u2_max": 15000.0,
}

# ============================================================
#  4. PARAMETER SAMPLING FUNCTION
# ============================================================
def sample_parameters(nominal, uncertainty, rng=None):
    """Sample a parameter dictionary from Gaussian distributions."""
    if rng is None:
        rng = np.random.default_rng()

    p = dict(nominal)
    for key in uncertainty:
        p[key] = rng.normal(nominal[key], uncertainty[key])

    # Physical constraints
    p["E_f"] = min(p["E_f"], 1.0)
    p["E_r"] = min(p["E_r"], 1.0)
    p["M"] = max(p["M"], 0.7 * nominal["M"])
    p["lf"] = np.clip(p["lf"], 0.3 * L_wb, 0.7 * L_wb)
    p["lr"] = L_wb - p["lf"]
    p["Jz"] = nominal["Jz"] * (p["M"] / nominal["M"])
    return p


# ============================================================
#  5. SINGLE TRAJECTORY SIMULATION
# ============================================================
def simulate_trajectory(params, v_base, sim_duration, dt, ctrl,
                        seg_curvatures, cum_lengths, total_length,
                        rng=None, excitation_strength="rich"):
    """Simulate one fixed-duration trajectory with richer input excitation.

    For identification data, completing a lap is not required. We only require
    that the run produces enough physically meaningful samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    L_wb_local = params["lf"] + params["lr"]

    t_max = float(sim_duration)
    N_steps = int(np.ceil(t_max / dt))

    # Rich excitation profiles are built before simulation because N_steps is known.
    exc = build_excitation_signals(N_steps, dt, v_base, rng, strength=excitation_strength)
    v_ref_profile = exc["v_ref_profile"]
    delta_add = exc["delta_add"]
    T_add = exc["T_add"]

    # Randomized initial condition improves coverage near but not exactly on the path.
    # s0 is randomized because fixed-duration runs may not complete a lap; this lets
    # the dataset cover both straight and curved track segments.
    s0 = rng.uniform(0.0, total_length)
    vx0 = float(np.clip(v_base + rng.normal(0.0, 0.8), 6.0, 34.0))
    y0 = rng.normal(0.0, 0.35)                       # m
    xi0 = rng.normal(0.0, 1.5 * np.pi / 180.0)       # rad
    vy0 = rng.normal(0.0, 0.20)                      # m/s
    omega0 = rng.normal(0.0, 0.025)                  # rad/s
    delta0 = rng.normal(0.0, 0.5 * np.pi / 180.0)    # rad
    Tr0 = rng.normal(0.0, 80.0)                      # N.m

    # State: [s, y, xi, vx, vy, omega, delta, Tr]
    x = np.array([s0, y0, xi0, vx0, vy0, omega0, delta0, Tr0])

    # Pre-allocate
    X_log = np.zeros((8, N_steps))
    Xdot_log = np.zeros((8, N_steps))
    U_log = np.zeros((2, N_steps))
    t_log = np.zeros(N_steps)
    alpha_log = np.zeros((2, N_steps))
    Ffc_log = np.zeros(N_steps)
    Frc_log = np.zeros(N_steps)
    vref_log = np.zeros(N_steps)
    delta_exc_log = np.zeros(N_steps)
    torque_exc_log = np.zeros(N_steps)

    k_end = 0
    lap_count = 0
    s_prev = 0.0

    for k in range(N_steps):
        t = k * dt
        s_cur = x[0]

        # Lap count is logged only for diagnostics. Do not stop at one lap.
        if np.floor(s_cur / total_length) > np.floor(s_prev / total_length):
            lap_count += 1
        s_prev = s_cur

        kap = get_curvature(s_cur, seg_curvatures, cum_lengths, total_length)

        # --- Steering: curvature feedforward + pure pursuit + injected excitation ---
        Ld = ctrl["Ld_min"] + ctrl["Kla"] * max(x[3], 1.0)
        kap_la = get_curvature(s_cur + Ld, seg_curvatures, cum_lengths, total_length)
        delta_ff = np.arctan(L_wb_local * kap_la)
        e_la = x[1] + Ld * np.sin(x[2])
        delta_fb = -np.arctan(2.0 * L_wb_local * e_la / Ld**2)
        delta_des = np.clip(
            delta_ff + delta_fb + delta_add[k],
            -ctrl["delta_max"], ctrl["delta_max"]
        )

        # --- Speed: time-varying reference + injected torque excitation ---
        v_ref_k = v_ref_profile[k]
        e_v = v_ref_k - x[3]
        T_des = np.clip(
            ctrl["Kp_speed"] * e_v + T_add[k],
            -ctrl["Tr_max"], ctrl["Tr_max"]
        )

        # --- Actuator-rate commands ---
        u1 = np.clip(ctrl["K_delta"] * (delta_des - x[6]), -ctrl["u1_max"], ctrl["u1_max"])
        u2 = np.clip(ctrl["K_T"] * (T_des - x[7]), -ctrl["u2_max"], ctrl["u2_max"])
        u = np.array([u1, u2])

        x_new, xdot, af, ar, Ff, Fr = rk4_step(
            x, u, params, dt, seg_curvatures, cum_lengths, total_length
        )

        k_end = k
        X_log[:, k] = x
        Xdot_log[:, k] = xdot
        U_log[:, k] = u
        t_log[k] = t
        alpha_log[:, k] = [af, ar]
        Ffc_log[k] = Ff
        Frc_log[k] = Fr
        vref_log[k] = v_ref_k
        delta_exc_log[k] = delta_add[k]
        torque_exc_log[k] = T_add[k]

        x = x_new

        # Safety abort: keep physically useful data only.
        if abs(x[1]) > 60.0 or x[3] < 1.0 or x[3] > 45.0 or abs(x[2]) > np.pi / 2:
            break

    n = k_end + 1
    return {
        "v_base": v_base,
        "X": X_log[:, :n],
        "Xdot": Xdot_log[:, :n],
        "U_rate": U_log[:, :n],
        "t": t_log[:n],
        "alpha": alpha_log[:, :n],
        "Ffc": Ffc_log[:n],
        "Frc": Frc_log[:n],
        "v_ref": vref_log[:n],
        "delta_exc": delta_exc_log[:n],
        "T_exc": torque_exc_log[:n],
        "n": n,
        "laps": lap_count,
    }


# ============================================================
#  6. MAIN GENERATION LOOP
# ============================================================
dt = 0.005
ds_factor = 6       # dt_eff = 0.03 s

# Rich-excitation identification runs are duration-based, not lap-based.
# The car does not need to complete a lap; it only needs to produce valid samples.
sim_duration_train = 16.0   # seconds per speed/parameter run
sim_duration_val = 16.0
sim_duration_test = 18.0

# Run/sample quality thresholds. These replace the old "must complete a lap" rule.
min_duration = 8.0
min_samples = int(np.ceil(min_duration / (dt * ds_factor)))

# Cap accepted runs with a contiguous fixed-dt window. This prevents a few long
# trajectories from dominating the regression while keeping each saved run usable
# as a real time sequence for rollout checks.
max_samples_per_run = 600
alpha_max = 20.0 * np.pi / 180.0
vy_max = 10.0
omega_max = 2.5
vx_min = 2.0
vx_max = 40.0
y_max = 60.0
xi_max = np.pi / 2

# More base speeds reduce dependence between vx, curvature, and omega.
v_bases_train = [12, 15, 18, 22, 25, 28]
v_bases_val = [14, 20, 26]
v_bases_test = [16, 23, 30]

# Richer excitation means fewer parameter trajectories can still give broad coverage.
N_traj_train = 120
N_traj_val = 8
N_traj_test = 10

rng = np.random.default_rng(42)


def longest_valid_segment(mask):
    """Return [start, end) positions of the longest contiguous True segment."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return None

    # Pad with False so rising/falling edges are easy to detect.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])


def choose_contiguous_window(valid_mask, min_len, max_len, rng):
    """
    Choose a contiguous window inside the longest valid segment.

    valid_mask is defined on the already-downsampled index array, so consecutive
    positions in this window are separated by exactly dt * ds_factor.
    """
    segment = longest_valid_segment(valid_mask)
    if segment is None:
        return None

    seg_start, seg_end = segment
    seg_len = seg_end - seg_start
    if seg_len < min_len:
        return None

    win_len = min(seg_len, max_len)
    if seg_len == win_len:
        win_start = seg_start
    else:
        win_start = int(rng.integers(seg_start, seg_end - win_len + 1))
    win_end = win_start + win_len
    return np.arange(win_start, win_end, dtype=int)


def generate_dataset(N_traj, label, rng, v_bases, sim_duration):
    """Generate N_traj parameter/track draws, each driven at multiple speed bases."""
    X_all = []
    Xd_all = []
    U_all = []
    param_records = []

    meta = {
        "traj_id": [],
        "run_id": [],
        "param_id": [],
        "v_base": [],
        "t": [],
        "R_turn": [],
        "L_straight": [],
        "v_ref": [],
        "delta_exc": [],
        "T_exc": [],
        "laps": [],
        "sim_duration": [],
    }

    run_counter = 0
    skipped = 0
    skip_reasons = {
        "too_short": 0,
        "too_few_valid_samples": 0,
        "nonfinite": 0,
    }

    for i in range(N_traj):
        if label == "val" and i == 0:
            p_i = dict(nominal)  # nominal validation trajectory for clean comparison
        else:
            p_i = sample_parameters(nominal, uncertainty, rng)

        param_record = {k: p_i[k] for k in [
            "B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r",
            "M", "Jz", "lf", "lr"
        ]}
        param_records.append(param_record)

        R_i, L_i, seg_curv_i, seg_len_i, cum_len_i, total_len_i = sample_track_geometry(rng, label)

        for v_base in v_bases:
            d = simulate_trajectory(
                p_i, v_base, sim_duration, dt, ctrl,
                seg_curv_i, cum_len_i, total_len_i,
                rng=rng, excitation_strength="rich"
            )

            idx = np.arange(0, d["n"], ds_factor)

            # Minimum-duration check: do not require a lap, but reject runs that
            # aborted too early to be useful for identification.
            if len(idx) < min_samples or d["t"][-1] < min_duration:
                skipped += 1
                skip_reasons["too_short"] += 1
                print(f"  WARNING: {label} traj={i}, v_base={v_base} too short; skipping.")
                continue

            X_run = np.column_stack([
                d["X"][3, idx],      # vx
                d["X"][4, idx],      # vy
                d["X"][5, idx],      # omega
                d["alpha"][0, idx],  # alpha_f
                d["alpha"][1, idx],  # alpha_r
            ])
            Xd_run = np.column_stack([
                d["Xdot"][3, idx],   # vx_dot
                d["Xdot"][4, idx],   # vy_dot
                d["Xdot"][5, idx],   # omega_dot
            ])
            U_run = np.column_stack([
                d["X"][6, idx],      # delta actual actuator state
                d["X"][7, idx],      # Tr actual actuator state
            ])

            finite_mask = (
                np.all(np.isfinite(X_run), axis=1)
                & np.all(np.isfinite(Xd_run), axis=1)
                & np.all(np.isfinite(U_run), axis=1)
            )
            if not np.any(finite_mask):
                skipped += 1
                skip_reasons["nonfinite"] += 1
                print(f"  WARNING: {label} traj={i}, v_base={v_base} nonfinite; skipping.")
                continue

            # Physical-validity sample filter. This removes impossible/unsafe points
            # without throwing away an otherwise informative excited trajectory.
            sample_valid = (
                finite_mask
                & (X_run[:, 0] > vx_min)
                & (X_run[:, 0] < vx_max)
                & (np.abs(X_run[:, 1]) < vy_max)
                & (np.abs(X_run[:, 2]) < omega_max)
                & (np.abs(X_run[:, 3]) < alpha_max)
                & (np.abs(X_run[:, 4]) < alpha_max)
                & (np.abs(U_run[:, 0]) <= ctrl["delta_max"] + 1e-9)
                & (np.abs(U_run[:, 1]) <= ctrl["Tr_max"] + 1e-9)
                & (np.abs(d["X"][1, idx]) < y_max)
                & (np.abs(d["X"][2, idx]) < xi_max)
            )

            # Keep a contiguous fixed-dt block of valid samples.
            # Do NOT simply do idx_valid = idx[sample_valid] here: that can create
            # gaps if isolated samples are filtered out. A contiguous block keeps
            # each accepted run time-consistent for rollout validation.
            window_pos = choose_contiguous_window(
                sample_valid,
                min_len=min_samples,
                max_len=max_samples_per_run,
                rng=rng,
            )

            if window_pos is None:
                skipped += 1
                skip_reasons["too_few_valid_samples"] += 1
                print(f"  WARNING: {label} traj={i}, v_base={v_base} has no valid contiguous window; skipping.")
                continue

            idx_valid = idx[window_pos]
            X_run = X_run[window_pos]
            Xd_run = Xd_run[window_pos]
            U_run = U_run[window_pos]
            n_i = X_run.shape[0]

            X_all.append(X_run)
            Xd_all.append(Xd_run)
            U_all.append(U_run)

            meta["traj_id"].append(np.full(n_i, i, dtype=int))
            meta["run_id"].append(np.full(n_i, run_counter, dtype=int))
            meta["param_id"].append(np.full(n_i, i, dtype=int))
            meta["v_base"].append(np.full(n_i, v_base, dtype=float))
            meta["t"].append(d["t"][idx_valid])
            meta["R_turn"].append(np.full(n_i, R_i, dtype=float))
            meta["L_straight"].append(np.full(n_i, L_i, dtype=float))
            meta["v_ref"].append(d["v_ref"][idx_valid])
            meta["delta_exc"].append(d["delta_exc"][idx_valid])
            meta["T_exc"].append(d["T_exc"][idx_valid])
            meta["laps"].append(np.full(n_i, d["laps"], dtype=int))
            meta["sim_duration"].append(np.full(n_i, sim_duration, dtype=float))

            run_counter += 1

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{label}] Completed parameter trajectory {i + 1}/{N_traj}")

    if not X_all:
        raise RuntimeError(f"No valid {label} trajectories were generated. Reduce excitation strength.")

    X_all = np.vstack(X_all)
    Xd_all = np.vstack(Xd_all)
    U_all = np.vstack(U_all)
    meta = {k: np.concatenate(v) for k, v in meta.items()}

    # Final sanity filter after stacking. The per-run filter already removed most
    # invalid samples; this is only a defensive check.
    valid = (
        np.all(np.isfinite(X_all), axis=1)
        & np.all(np.isfinite(Xd_all), axis=1)
        & np.all(np.isfinite(U_all), axis=1)
        & (X_all[:, 0] > vx_min)
        & (X_all[:, 0] < vx_max)
    )
    X_all = X_all[valid]
    Xd_all = Xd_all[valid]
    U_all = U_all[valid]
    meta = {k: v[valid] for k, v in meta.items()}

    print(f"  [{label}] skipped runs: {skipped}")
    print(f"  [{label}] skip reasons: {skip_reasons}")
    return X_all, Xd_all, U_all, param_records, meta


print("\n=== Generating TRAINING data with rich excitation ===")
X_train, Xd_train, U_train, params_train, meta_train = generate_dataset(
    N_traj_train, "train", rng, v_bases_train, sim_duration_train
)

print("\n=== Generating VALIDATION data with rich excitation ===")
X_val, Xd_val, U_val, params_val, meta_val = generate_dataset(
    N_traj_val, "val", rng, v_bases_val, sim_duration_val
)

print("\n=== Generating TEST data with rich excitation ===")
X_test, Xd_test, U_test, params_test, meta_test = generate_dataset(
    N_traj_test, "test", rng, v_bases_test, sim_duration_test
)


# ============================================================
#  7. SAVE
# ============================================================
out_dir = Path(__file__).resolve().parent / "Data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "vindy_data_rich_excitation.npz"


def params_to_arrays(records):
    keys = list(records[0].keys())
    return {k: np.array([r[k] for r in records]) for k in keys}


params_train_arr = params_to_arrays(params_train)
params_val_arr = params_to_arrays(params_val)
params_test_arr = params_to_arrays(params_test)

save_dict = {
    "X_train": X_train, "Xd_train": Xd_train, "U_train": U_train,
    "X_val": X_val, "Xd_val": Xd_val, "U_val": U_val,
    "X_test": X_test, "Xd_test": Xd_test, "U_test": U_test,
}

save_dict.update({f"ptrain_{k}": v for k, v in params_train_arr.items()})
save_dict.update({f"pval_{k}": v for k, v in params_val_arr.items()})
save_dict.update({f"ptest_{k}": v for k, v in params_test_arr.items()})
save_dict.update({f"nom_{k}": np.array(nominal[k]) for k in [
    "B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r",
    "M", "Jz", "lf", "lr", "R_wheel"
]})
save_dict.update({f"train_{k}": v for k, v in meta_train.items()})
save_dict.update({f"val_{k}": v for k, v in meta_val.items()})
save_dict.update({f"test_{k}": v for k, v in meta_test.items()})

np.savez(out_path, **save_dict)

print(f"\n=== VINDy Dataset Saved to {out_path} ===")
print(f"  Training:   {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Test:       {X_test.shape[0]} samples")
print("  X:    [vx, vy, omega, alpha_f, alpha_r]")
print("  Xdot: [vx_dot, vy_dot, omega_dot]")
print("  U:    [delta, Tr]")
print(f"  Downsample factor: {ds_factor}  (dt_eff = {dt * ds_factor:.3f} s)")
print(f"  Duration-based runs: train={sim_duration_train:.1f}s, val={sim_duration_val:.1f}s, test={sim_duration_test:.1f}s")
print(f"  Minimum accepted duration: {min_duration:.1f}s  ({min_samples} downsampled samples)")
print(f"  Max samples per accepted run: {max_samples_per_run} contiguous samples")
print(f"  Physical filters: |alpha|<{alpha_max * 180 / np.pi:.1f} deg, |vy|<{vy_max}, |omega|<{omega_max}")
print(f"  N_traj_train = {N_traj_train},  N_traj_val = {N_traj_val},  N_traj_test = {N_traj_test}")


# ============================================================
#  8. DIAGNOSTIC PLOTS
# ============================================================
plot_dir = Path(__file__).resolve().parent / "Plots"
plot_dir.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    path = plot_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# --- 8a. Parameter distributions ---
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Sampled Parameter Distributions (Training)", fontsize=14)

param_keys = ["B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r", "M", "Jz", "lf", "lr"]
for ax, key in zip(axes.flat, param_keys):
    vals = params_train_arr[key]
    ax.hist(vals, bins=20, edgecolor="k", alpha=0.7)
    ax.axvline(nominal.get(key, vals.mean()), color="r", ls="--", lw=2, label="nominal")
    ax.set_title(key)
    ax.legend(fontsize=7)
fig.tight_layout()
save_fig(fig, "vindy_rich_param_distributions.png")


# --- 8b. Slip angle coverage ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Slip Angle Coverage (Rich Training Data)")
axes[0].hist(X_train[:, 3] * 180 / np.pi, bins=100, edgecolor="k", alpha=0.7)
axes[0].set_xlabel("alpha_f [deg]")
axes[0].set_title("Front slip angle")
axes[0].grid(True, alpha=0.3)
axes[1].hist(X_train[:, 4] * 180 / np.pi, bins=100, edgecolor="k", alpha=0.7)
axes[1].set_xlabel("alpha_r [deg]")
axes[1].set_title("Rear slip angle")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "vindy_rich_slip_angle_coverage.png")


# --- 8c. Input coverage ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Input Coverage (Rich Training Data)")
axes[0].hist(U_train[:, 0] * 180 / np.pi, bins=100, edgecolor="k", alpha=0.7)
axes[0].set_xlabel("delta [deg]")
axes[0].set_title("Steering actuator state")
axes[0].grid(True, alpha=0.3)
axes[1].hist(U_train[:, 1], bins=100, edgecolor="k", alpha=0.7)
axes[1].set_xlabel("Tr [N.m]")
axes[1].set_title("Wheel torque actuator state")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "vindy_rich_input_coverage.png")


# --- 8d. Derivative distributions ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Derivative Distributions (Rich Training Data)")
dot_names = [r"$\dot{v}_x$", r"$\dot{v}_y$", r"$\dot{\omega}$"]
for ax, col, name in zip(axes, range(3), dot_names):
    ax.hist(Xd_train[:, col], bins=100, edgecolor="k", alpha=0.7)
    ax.set_xlabel(name)
    ax.set_title(f"{name} distribution")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "vindy_rich_derivative_distributions.png")


# --- 8e. State statistics ---
print("\n--- Training Data Statistics ---")
state_names = ["vx", "vy", "omega", "alpha_f", "alpha_r"]
for i, name in enumerate(state_names):
    vals = X_train[:, i]
    scale = 180 / np.pi if "alpha" in name else 1.0
    unit = "deg" if "alpha" in name else ("[m/s]" if "v" in name else "[rad/s]")
    print(f"  {name:8s}: min={vals.min() * scale:+8.3f}, max={vals.max() * scale:+8.3f}, "
          f"mean={vals.mean() * scale:+8.3f}, std={vals.std() * scale:8.3f} {unit}")

print("\n--- Input Statistics ---")
input_stats = [("delta", U_train[:, 0] * 180 / np.pi, "deg"), ("Tr", U_train[:, 1], "N.m")]
for name, vals, unit in input_stats:
    print(f"  {name:8s}: min={vals.min():+10.3f}, max={vals.max():+10.3f}, "
          f"mean={vals.mean():+10.3f}, std={vals.std():10.3f} {unit}")

print("\n--- Derivative Statistics ---")
dot_labels = ["vx_dot", "vy_dot", "omega_dot"]
for i, name in enumerate(dot_labels):
    vals = Xd_train[:, i]
    print(f"  {name:10s}: min={vals.min():+10.4f}, max={vals.max():+10.4f}, "
          f"mean={vals.mean():+10.4f}, std={vals.std():10.4f}")

print("\nDone.")
