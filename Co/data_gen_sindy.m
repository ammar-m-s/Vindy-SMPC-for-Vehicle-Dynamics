%% DATA_GEN_SINDY  Data generation for SINDy vehicle dynamics identification.
%
% Drives a 3-DOF single-track vehicle model (Eq. 9) with Pacejka tire 
% forces (Eq. 11) around an oval track at multiple reference speeds using 
% curvature feedforward + pure pursuit (steering) and P speed control.
%
% Outputs:
%   sindy_data.mat  - assembled SINDy dataset
%   Diagnostic plots: trajectories, velocities, slip angles, tire forces

clear; clc; close all;

%% ============================================================
%  1. VEHICLE PARAMETERS (Table 2)
%  ============================================================
params.M       = 1412;              % vehicle mass [kg]
params.Jz      = 1536.7;            % yaw moment of inertia [kg*m^2]
params.R_wheel = 0.325;             % wheel radius [m]
params.lf      = 1.015;             % CG to front axle [m]
params.lr      = 1.895;             % CG to rear axle [m]
params.B_pac   = 0.0885 * (180/pi); % Pacejka B, converted deg^-1 -> rad^-1
params.C_pac   = 1.4;               % Pacejka C (shape)
params.D_pac   = 8311;              % Pacejka D (peak force) [N]
params.E_pac   = -2;                % Pacejka E (asymmetry)
params.Cr      = 0.015;             % rolling resistance coefficient
params.rho     = 1.225;             % air density [kg/m^3]
params.Cd      = 0.3;               % drag coefficient
params.Af      = 2.2;               % frontal area [m^2]
params.g       = 9.81;              % gravity [m/s^2]

%% ============================================================
%  2. CONTROLLER GAINS
%  ============================================================
ctrl.Ld_min    = 8;              % min lookahead distance [m]
ctrl.Kla       = 0.6;            % speed-dependent lookahead gain [s]
ctrl.Kp_speed  = 3000;           % P speed gain
ctrl.K_delta   = 10;             % steering rate tracking gain
ctrl.K_T       = 15;             % torque rate tracking gain
ctrl.delta_max = 30 * pi/180;    % max steering angle [rad]
ctrl.Tr_max    = 5000;           % max torque [Nm]
ctrl.u1_max    = 2.0;            % max steering rate [rad/s]
ctrl.u2_max    = 15000;          % max torque rate [Nm/s]

%% ============================================================
%  3. GENERATE TRACK
%  ============================================================
% Oval: two straights connected by two 180-deg turns
% R=80m ensures all speeds up to ~25 m/s are within tire friction limits
R_turn     = 80;
L_straight = 150;

seg_curvatures = [0;  1/R_turn;  0;  1/R_turn];
seg_lengths    = [L_straight;  pi*R_turn;  L_straight;  pi*R_turn];
cum_lengths    = cumsum(seg_lengths);
total_length   = cum_lengths(end);

% Wheelbase
L_wb = params.lf + params.lr;

% Max lateral accel check
a_lat_max = 2 * params.D_pac / params.M;  % peak tire force both axles
v_max_turn = sqrt(a_lat_max * R_turn);
fprintf('Track: %.0f m total (straights=%.0f m, turns R=%.0f m)\n', total_length, L_straight, R_turn);
fprintf('Max feasible turn speed: %.1f m/s (a_lat_max=%.1f m/s^2)\n\n', v_max_turn, a_lat_max);

% Precompute centerline for plotting
ds_track = 0.1;
s_vec    = (0:ds_track:total_length)';
N_pts    = length(s_vec);
X_center = zeros(N_pts,1);
Y_center = zeros(N_pts,1);
theta_c  = zeros(N_pts,1);
theta    = 0;
for i = 2:N_pts
    kap    = get_curvature(s_vec(i), seg_curvatures, cum_lengths, total_length);
    theta  = theta + kap * ds_track;
    theta_c(i) = theta;
    X_center(i) = X_center(i-1) + cos(theta) * ds_track;
    Y_center(i) = Y_center(i-1) + sin(theta) * ds_track;
end
half_width = 8;  % track half-width [m]

%% ============================================================
%  4. SIMULATION
%  ============================================================
dt     = 0.001;                 % integration timestep [s]
v_refs = [10, 15, 20, 25];     % reference speeds [m/s]
n_laps = 2;

all_data = struct();

for iv = 1:length(v_refs)
    v_ref = v_refs(iv);
    fprintf('--- v_ref = %d m/s ---\n', v_ref);
    
    % Required lateral accel in turns
    a_req = v_ref^2 / R_turn;
    fprintf('  Required turn a_lat: %.1f m/s^2 (max=%.1f)\n', a_req, a_lat_max);
    
    % Initial state: [s, y, xi, vx, vy, omega, delta, Tr]
    Fres_init = params.Cr*params.M*params.g + 0.5*params.rho*params.Cd*params.Af*v_ref^2;
    Tr_init   = Fres_init * params.R_wheel;
    x = [0; 0; 0; v_ref; 0; 0; 0; Tr_init];
    
    % Generous simulation time
    t_max   = n_laps * total_length / max(v_ref, 3) * 2.0;
    N_steps = ceil(t_max / dt);
    
    % Preallocate
    X_log     = zeros(8, N_steps);
    Xdot_log  = zeros(8, N_steps);
    U_log     = zeros(2, N_steps);
    t_log     = zeros(1, N_steps);
    alpha_log = zeros(2, N_steps);
    Ffc_log   = zeros(1, N_steps);
    Frc_log   = zeros(1, N_steps);
    
    k_end       = 0;
    lap_count   = 0;
    s_prev      = 0;
    
    for k = 1:N_steps
        t = (k-1) * dt;
        s_cur = x(1);
        
        % --- Detect lap completion: s crosses multiple of total_length ---
        if floor(s_cur / total_length) > floor(s_prev / total_length)
            lap_count = lap_count + 1;
            fprintf('  Lap %d completed at t = %.2f s\n', lap_count, t);
            if lap_count >= n_laps
                k_end = k;
                break;
            end
        end
        s_prev = s_cur;
        
        % Road curvature
        kap = get_curvature(s_cur, seg_curvatures, cum_lengths, total_length);
        
        % --- STEERING: curvature feedforward + pure pursuit feedback ---
        Ld = ctrl.Ld_min + ctrl.Kla * max(x(4), 1);
        
        % Feedforward: steer for upcoming road curvature
        kap_la   = get_curvature(s_cur + Ld, seg_curvatures, cum_lengths, total_length);
        delta_ff = atan(L_wb * kap_la);
        
        % Feedback: correct lateral and heading errors
        e_la     = x(2) + Ld * sin(x(3));
        delta_fb = -atan(2 * L_wb * e_la / Ld^2);
        
        delta_des = clamp(delta_ff + delta_fb, -ctrl.delta_max, ctrl.delta_max);
        
        % --- SPEED: feedforward (overcome drag) + proportional feedback ---
        Fres_now = params.Cr*params.M*params.g + 0.5*params.rho*params.Cd*params.Af*x(4)^2;
        Tr_ff    = Fres_now * params.R_wheel;   % torque needed just to maintain current speed
        e_v      = v_ref - x(4);
        T_des    = clamp(Tr_ff + ctrl.Kp_speed * e_v, -ctrl.Tr_max, ctrl.Tr_max);
        
        % --- Rate inputs ---
        u1 = clamp(ctrl.K_delta * (delta_des - x(7)), -ctrl.u1_max, ctrl.u1_max);
        u2 = clamp(ctrl.K_T     * (T_des     - x(8)), -ctrl.u2_max, ctrl.u2_max);
        u  = [u1; u2];
        
        % --- Compute derivatives and log ---
        [xdot, af, ar, Ff, Fr] = vehicle_ode(x, u, params, kap);
        
        k_end          = k;
        X_log(:,k)     = x;
        Xdot_log(:,k)  = xdot;
        U_log(:,k)     = u;
        t_log(k)       = t;
        alpha_log(:,k) = [af; ar];
        Ffc_log(k)     = Ff;
        Frc_log(k)     = Fr;
        
        % --- RK4 integration ---
        k1 = xdot;
        k2 = vehicle_ode(x + dt/2*k1, u, params, ...
             get_curvature(x(1)+dt/2*k1(1), seg_curvatures, cum_lengths, total_length));
        k3 = vehicle_ode(x + dt/2*k2, u, params, ...
             get_curvature(x(1)+dt/2*k2(1), seg_curvatures, cum_lengths, total_length));
        k4 = vehicle_ode(x + dt*k3, u, params, ...
             get_curvature(x(1)+dt*k3(1), seg_curvatures, cum_lengths, total_length));
        x  = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
        
        % --- Safety: abort if vehicle totally lost ---
        if abs(x(2)) > 50 || x(4) < 0.1
            fprintf('  ABORT at t=%.2f s: y=%.1f m, vx=%.2f m/s\n', t, x(2), x(4));
            break;
        end
    end
    
    % Trim and store
    all_data(iv).v_ref   = v_ref;
    all_data(iv).X       = X_log(:,1:k_end);
    all_data(iv).Xdot    = Xdot_log(:,1:k_end);
    all_data(iv).U       = U_log(:,1:k_end);
    all_data(iv).t       = t_log(1:k_end);
    all_data(iv).alpha   = alpha_log(:,1:k_end);
    all_data(iv).Ffc     = Ffc_log(1:k_end);
    all_data(iv).Frc     = Frc_log(1:k_end);
    all_data(iv).n       = k_end;
    all_data(iv).laps    = lap_count;
    fprintf('  %d laps, %d samples (%.1f s)\n\n', lap_count, k_end, t_log(k_end));
end

%% ============================================================
%  5. ASSEMBLE SINDy DATASET
%  ============================================================
ds_factor  = 10;   % downsample: keep every 10th point
X_sindy    = [];
Xdot_sindy = [];
U_sindy    = [];

for iv = 1:length(v_refs)
    if all_data(iv).laps < 1
        fprintf('WARNING: v_ref=%d did not complete a lap, skipping.\n', v_refs(iv));
        continue;
    end
    idx = 1:ds_factor:all_data(iv).n;
    X_sindy    = [X_sindy;    all_data(iv).X(4,idx)',  all_data(iv).X(5,idx)', ...
                               all_data(iv).X(6,idx)',  all_data(iv).alpha(1,idx)', ...
                               all_data(iv).alpha(2,idx)'];
    Xdot_sindy = [Xdot_sindy; all_data(iv).Xdot(4,idx)', all_data(iv).Xdot(5,idx)', ...
                               all_data(iv).Xdot(6,idx)'];
    U_sindy    = [U_sindy;    all_data(iv).X(7,idx)', all_data(iv).X(8,idx)'];
end

fprintf('=== SINDy Dataset ===\n');
fprintf('  %d samples\n', size(X_sindy,1));
fprintf('  X:    [vx, vy, omega, alpha_f, alpha_r]\n');
fprintf('  Xdot: [vx_dot, vy_dot, omega_dot]\n');
fprintf('  U:    [delta, Tr]\n');

save('sindy_data.mat', 'X_sindy', 'Xdot_sindy', 'U_sindy', 'all_data', 'params');
fprintf('  Saved to sindy_data.mat\n\n');

%% ============================================================
%  6. PLOTS
%  ============================================================
colors = lines(length(v_refs));

% --- Trajectories: one subplot per speed ---
figure('Name','Trajectories','Position',[100 100 1200 900]);
X_in  = X_center + half_width*sin(theta_c);
Y_in  = Y_center - half_width*cos(theta_c);
X_out = X_center - half_width*sin(theta_c);
Y_out = Y_center + half_width*cos(theta_c);

for iv = 1:length(v_refs)
    subplot(2, 2, iv); hold on;
    plot(X_in, Y_in, 'k--', 'LineWidth', 1, 'DisplayName', 'Boundaries');
    plot(X_out, Y_out, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    plot(X_center, Y_center, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Reference');
    
    % Convert curvilinear to global
    pidx = 1:100:all_data(iv).n;
    s_p  = all_data(iv).X(1, pidx);
    y_p  = all_data(iv).X(2, pidx);
    s_w  = mod(s_p, total_length);
    Xc_i = interp1(s_vec, X_center, s_w, 'linear', 'extrap');
    Yc_i = interp1(s_vec, Y_center, s_w, 'linear', 'extrap');
    tc_i = interp1(s_vec, theta_c,  s_w, 'linear', 'extrap');
    Xg   = Xc_i - y_p .* sin(tc_i);
    Yg   = Yc_i + y_p .* cos(tc_i);
    plot(Xg, Yg, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Actual');
    
    axis equal; grid on;
    xlabel('X [m]'); ylabel('Y [m]');
    title(sprintf('v_{ref} = %d m/s  (%d laps)', v_refs(iv), all_data(iv).laps));
    legend('Location', 'best');
end
sgtitle('Reference vs Actual Trajectories');

% --- Velocities ---
figure('Name','Velocities','Position',[100 100 900 700]);
labels = {'v_x [m/s]','v_y [m/s]','\omega [deg/s]'};
rows   = [4, 5, 6];
scales = [1, 1, 180/pi];
for j = 1:3
    subplot(3,1,j); hold on;
    for iv = 1:length(v_refs)
        plot(all_data(iv).t, all_data(iv).X(rows(j),:)*scales(j), ...
             'Color', colors(iv,:), 'LineWidth', 1, ...
             'DisplayName', sprintf('%d m/s', v_refs(iv)));
    end
    ylabel(labels{j}); grid on;
    if j==1, legend('Location','best'); title('Velocity Profiles'); end
    if j==3, xlabel('Time [s]'); end
end

% --- Slip angles ---
figure('Name','Slip Angles','Position',[100 100 900 500]);
for j = 1:2
    subplot(2,1,j); hold on;
    for iv = 1:length(v_refs)
        plot(all_data(iv).t, all_data(iv).alpha(j,:)*180/pi, ...
             'Color', colors(iv,:), 'LineWidth', 1, ...
             'DisplayName', sprintf('%d m/s', v_refs(iv)));
    end
    if j==1, ylabel('\alpha_f [deg]'); title('Slip Angles'); legend('Location','best');
    else,    ylabel('\alpha_r [deg]'); xlabel('Time [s]'); end
    grid on;
end

% --- Tire forces vs slip angle ---
figure('Name','Tire Forces','Position',[100 100 900 500]);
a_range = linspace(-20,20,500)*pi/180;
F_ref   = pacejka(a_range, params.B_pac, params.C_pac, params.D_pac, params.E_pac);

subplot(1,2,1);
plot(a_range*180/pi, F_ref, 'k-', 'LineWidth', 2, 'DisplayName','Pacejka'); hold on;
for iv = 1:length(v_refs)
    pidx = 1:200:all_data(iv).n;
    scatter(all_data(iv).alpha(1,pidx)*180/pi, all_data(iv).Ffc(pidx), ...
            5, colors(iv,:), 'filled', 'DisplayName', sprintf('%d m/s',v_refs(iv)));
end
xlabel('\alpha_f [deg]'); ylabel('F_{fc} [N]');
title('Front Lateral Force'); legend('Location','best'); grid on;

subplot(1,2,2);
plot(a_range*180/pi, F_ref, 'k-', 'LineWidth', 2, 'DisplayName','Pacejka'); hold on;
for iv = 1:length(v_refs)
    pidx = 1:200:all_data(iv).n;
    scatter(all_data(iv).alpha(2,pidx)*180/pi, all_data(iv).Frc(pidx), ...
            5, colors(iv,:), 'filled', 'DisplayName', sprintf('%d m/s',v_refs(iv)));
end
xlabel('\alpha_r [deg]'); ylabel('F_{rc} [N]');
title('Rear Lateral Force'); legend('Location','best'); grid on;

% --- Control inputs ---
figure('Name','Inputs','Position',[100 100 900 500]);
subplot(2,1,1); hold on;
for iv = 1:length(v_refs)
    plot(all_data(iv).t, all_data(iv).X(7,:)*180/pi, 'Color',colors(iv,:),'LineWidth',1, ...
         'DisplayName',sprintf('%d m/s',v_refs(iv)));
end
ylabel('\delta [deg]'); title('Control Inputs'); legend('Location','best'); grid on;
subplot(2,1,2); hold on;
for iv = 1:length(v_refs)
    plot(all_data(iv).t, all_data(iv).X(8,:), 'Color',colors(iv,:),'LineWidth',1);
end
ylabel('T_r [Nm]'); xlabel('Time [s]'); grid on;

% --- Slip angle statistics ---
fprintf('--- Slip Angle Coverage ---\n');
for iv = 1:length(v_refs)
    af = all_data(iv).alpha(1,:)*180/pi;
    ar = all_data(iv).alpha(2,:)*180/pi;
    fprintf('  %2d m/s (%d laps):  alpha_f=[%+.1f, %+.1f] deg   alpha_r=[%+.1f, %+.1f] deg\n', ...
            v_refs(iv), all_data(iv).laps, min(af), max(af), min(ar), max(ar));
end

%% ============================================================
%  LOCAL FUNCTIONS
%  ============================================================

function [xdot, alpha_f, alpha_r, Ffc, Frc] = vehicle_ode(x, u, p, kappa)
% 3-DOF single-track vehicle model (Equation 9)
    vx = x(4); vy = x(5); omega = x(6); delta = x(7); Tr = x(8);
    
    % Slip angles (Equation 10)
    vy_front  = vy + omega * p.lf;
    v_lat_w   = vy_front*cos(delta) - vx*sin(delta);
    v_lon_w   = vx*cos(delta) + vy_front*sin(delta);
    alpha_f   = atan2(v_lat_w, max(abs(v_lon_w), 0.5));
    alpha_r   = atan2(vy - omega*p.lr, max(abs(vx), 0.5));
    
    % Pacejka lateral forces (Equation 11)
    Ffc = pacejka(alpha_f, p.B_pac, p.C_pac, p.D_pac, p.E_pac);
    Frc = pacejka(alpha_r, p.B_pac, p.C_pac, p.D_pac, p.E_pac);
    
    % Longitudinal force and resistance
    Frl  = Tr / p.R_wheel;
    Fres = p.Cr*p.M*p.g + 0.5*p.rho*p.Cd*p.Af*vx^2;
    
    % Kinematic equations (protect against singularity when y*kappa -> 1)
    s_dot  = (vx*cos(x(3)) - vy*sin(x(3))) / max(1 - x(2)*kappa, 0.01);
    y_dot  = vx*sin(x(3)) + vy*cos(x(3));
    xi_dot = omega - kappa*s_dot;
    
    % Dynamic equations (what SINDy will identify)
    vx_dot    = omega*vy + Frl/p.M - Ffc*sin(delta)/p.M - Fres/p.M;
    vy_dot    = -omega*vx + Frc/p.M + Ffc*cos(delta)/p.M;
    omega_dot = (1/p.Jz) * (-Frc*p.lr + Ffc*p.lf*cos(delta));
    
    xdot = [s_dot; y_dot; xi_dot; vx_dot; vy_dot; omega_dot; u(1); u(2)];
end

function F = pacejka(alpha, B, C, D, E)
    Ba = B .* alpha;
    F  = D .* sin(C .* atan(Ba - E.*(Ba - atan(Ba))));
end

function kappa = get_curvature(s, seg_curvatures, cum_lengths, total_length)
    s = mod(s, total_length);
    for i = 1:length(seg_curvatures)
        if s < cum_lengths(i)
            kappa = seg_curvatures(i);
            return;
        end
    end
    kappa = seg_curvatures(end);
end

function y = clamp(x, lo, hi)
    y = max(lo, min(hi, x));
end
