function dxdt = ring_model(t, x, W, tau, I)
% RING_MODEL   Defines the rate-based ring attractor dynamics
% dx/dt = (-x + W * phi(x)/N + I) / tau

    N = length(x);
    phi = @(x) (1 + tanh(x)) / 2;      % Activation function

    dxdt = (-x + (W * phi(x)) / N + I) / tau;
end
