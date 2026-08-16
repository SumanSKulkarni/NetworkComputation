%% ring_attractor_landscape.m

N = 50;
tau = 0.01;                        % time constant
J_E = 10; J_I = -9;
I = 1 * ones(N, 1);

dt = 0.01;
T_total = 1;
tspan = 0:dt:T_total;
phi = @(x) (1 + tanh(x)) / 2;

theta = linspace(0, 2*pi, N+1); theta(end) = [];

W = zeros(N);
for j = 1:N
    for k = 1:N
        W(j, k) = J_I + J_E * cos(theta(j) - theta(k));
    end
end

% Small Gaussian noise on the connectivity
noise_std = 0.4;
noise = noise_std * randn(N);
noise_sym = (noise + noise') / 2;
W = W + noise_sym;

%% Simulate to find the stable bump

bump_center = N/2 - 4.3;
sigma = N / 20;
x0 = exp(-((1:N)' - bump_center).^2 / (2 * sigma^2));

[~, x_sol] = ode45(@(t, x) ring_model(t, x, W, tau, I), tspan, x0);
x_stable = x_sol(end, :)';
bump_rate = phi(x_stable);

n_show = 8;
shifts = round(linspace(0, N, n_show + 1));
shifts(end) = [];
colors_c = crameri('batlow', n_show);

figure; hold on;
for i = 1:n_show
    bump_i = circshift(bump_rate, shifts(i));
    plot(theta, bump_i, 'Color', colors_c(i, :), 'LineWidth', 1.5);
end
xlabel('Heading angle \theta'); ylabel('Firing rate');
xticks([0 pi 2*pi]); xticklabels({'0', '\pi', '2\pi'});
title('Panel (c): stable bumps at different headings');
box off;

%% Linearize about the stable bump (Jacobian)

dphi = @(x) 0.5 * (1 - tanh(x).^2);
D_phi = diag(dphi(x_stable));
A = -eye(N) + (W * D_phi) / N;

eigvals_A = eig(A);

figure;
plot(real(eigvals_A), imag(eigvals_A), 'ko', 'MarkerFaceColor', [0.2 0.4 0.7]);
xline(0, ':', 'Color', [0.5 0.5 0.5]);
xlabel('Real part'); ylabel('Imaginary part');
title('Eigenvalues of the Jacobian A at the stable bump');
grid on; axis equal;

%% Compute the Gramian 

B = eye(N);                        % independent input to every neuron
T_horizon = 1;

[~, Wc] = control_energy(A, B, zeros(N, 1), zeros(N, 1), T_horizon);

[V_W, D_W] = eig(Wc);
lambda_W = diag(D_W);
[lambda_sorted, idx] = sort(lambda_W, 'descend');   % descending Gramian eigenvalue
V_W_sorted = V_W(:, idx);                           % => ascending cost

eps_reg = 1e-12;
inv_lambda = 1 ./ max(lambda_sorted, eps_reg);      % cost eigenvalues, cheapest first

figure; hold on;
for k = 1:numel(inv_lambda)
    line([inv_lambda(k) inv_lambda(k)], [0 1], 'Color', 'k', 'LineWidth', 1.2);
end
set(gca, 'XScale', 'log');
decade_min = floor(log10(min(inv_lambda)));
decade_max = ceil(log10(max(inv_lambda)));
tick_vals = 10.^(decade_min:decade_max);
set(gca, 'XTick', tick_vals);
set(gca, 'XTickLabel', arrayfun(@(e) sprintf('10^{%d}', e), ...
    decade_min:decade_max, 'UniformOutput', false));
xlim([tick_vals(1) tick_vals(end)]);
ylim([-0.15 1.4]);
set(gca, 'YTick', [], 'YColor', 'none');
box off;

cheap_val = inv_lambda(1);
plot(cheap_val, 1.15, 'v', 'MarkerFaceColor', [0.85 0.55 0.13], ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 10);
xlabel('1 / \lambda_i(W_c)');
title('Panel (d): affordance landscape eigenspectrum (least-costly mode marked)');

%% Bump derivative and least-costly mode

d_bump_rate = circshift(bump_rate, -1) - circshift(bump_rate, 1); 
d_bump_rate = d_bump_rate / norm(d_bump_rate);

scale_to_bump = @(v) v * (max(bump_rate) / max(abs(v)));

v_cheap_h = V_W_sorted(:, 1) / norm(V_W_sorted(:, 1));
v_cheap_rate = D_phi * v_cheap_h;
v_cheap_rate = v_cheap_rate / norm(v_cheap_rate);
alignment_rate = dot(v_cheap_rate, d_bump_rate);
v_cheap_rate_s = scale_to_bump(v_cheap_rate);

fprintf('Alignment (cosine) between cheapest mode and bump derivative: %.3f\n', alignment_rate);

%% Second-cheapest and most-expensive modes (SI. Fig. S2)

v_cheap2_h = V_W_sorted(:, 2) / norm(V_W_sorted(:, 2));
v_cheap2_rate = D_phi * v_cheap2_h;
v_cheap2_rate = v_cheap2_rate / norm(v_cheap2_rate);
v_cheap2_rate_s = scale_to_bump(v_cheap2_rate);

v_exp_h = V_W_sorted(:, end) / norm(V_W_sorted(:, end));
v_exp_rate = D_phi * v_exp_h;
v_exp_rate = v_exp_rate / norm(v_exp_rate);
v_exp_rate_s = scale_to_bump(v_exp_rate);

figure('Position', [100 100 500 750]);

subplot(3, 1, 1); hold on;
plot(theta, bump_rate, 'Color', [0.6 0.6 0.6], 'LineWidth', 2.2);
plot(theta, v_cheap_rate_s, 'Color', [0.85 0.55 0.13], 'LineWidth', 2.2);
plot(theta, d_bump_rate * (max(bump_rate)/max(abs(d_bump_rate))), '--', ...
    'Color', [0.15 0.15 0.65], 'LineWidth', 2.0);
ylabel('Activity');
legend('Stable activity h^*', 'Least costly mode', '\partial h^*/\partial\theta', ...
    'Location', 'best');
xticks([0 pi 2*pi]); xticklabels({'0', '\pi', '2\pi'});
title(sprintf('Least costly mode (w_1 = %.2g, alignment = %.2f)', ...
    1/lambda_sorted(1), alignment_rate));
box off;

subplot(3, 1, 2); hold on;
plot(theta, bump_rate, 'Color', [0.6 0.6 0.6], 'LineWidth', 2.2);
plot(theta, v_cheap2_rate_s, 'Color', [0.85 0.55 0.13], 'LineWidth', 2.2);
ylabel('Activity');
legend('Stable activity h^*', 'Second least costly mode', 'Location', 'best');
xticks([0 pi 2*pi]); xticklabels({'0', '\pi', '2\pi'});
title(sprintf('Second-cheapest mode (w_2 = %.2g)', 1/lambda_sorted(2)));
box off;

subplot(3, 1, 3); hold on;
plot(theta, bump_rate, 'Color', [0.6 0.6 0.6], 'LineWidth', 2.2);
plot(theta, v_exp_rate_s, 'Color', [0.55 0.13 0.55], 'LineWidth', 2.2);
xlabel('Heading angle \theta'); ylabel('Activity');
legend('Stable activity h^*', 'Most expensive mode', 'Location', 'best');
xticks([0 pi 2*pi]); xticklabels({'0', '\pi', '2\pi'});
title(sprintf('Most expensive mode (w_N = %.2g)', 1/lambda_sorted(end)));
box off;

sgtitle('Panel (e) / Supp. Fig. S2: affordance landscape modes');