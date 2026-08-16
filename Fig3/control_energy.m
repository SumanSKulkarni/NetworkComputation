function [E,Wc,x_t,u_t] = control_energy(A, B, x0, xf, T)
% Computes minimum control energy for state transition.
% A: System adjacency matrix:       n x n
% B: Control input matrix 
% x0: Initial state:                n x 1
% xf: Final state:                  n x 1
% T: Control horizon
% 
% Outputs
% E: Minimum control energy

% Normalize
c = 0;
%A = A / abs(max(eigs(A,1)+1))  - c*eye(size(A));
%A_norm = -1* A / max(eigs(A,1));
% State transition to achieve
Phi = xf-expm(A*T)*x0;
% Gramian
Wc = integral(@(t)(expm(A*t)*B)*(expm(A*t)*B)', 0, T, 'ArrayValued', 1,...
              'AbsTol',1e-16,'RelTol',1e-12);
% Inverse
WcI = Wc^-1;
%WcI = pinv(Wc);
% Energy
E = sum((WcI*Phi).*Phi);

N = length(x0);
x_t = zeros(N,1); u_t = zeros(N,1);

% Time discretization
m = size(B,2);
L = 1000;
t_vec = linspace(0, T, L);
dt = T / (L - 1);

BT = B';
AtT = A';

% Compute u(t) over time
u_t = zeros(m, L);
for i = 1:L
    t = t_vec(i);
    u_t(:, i) = BT * expm(AtT*(T - t)) * WcI * Phi;
end
%x_t = zeros(size(x0,1), L);
%for i = 1:L
%    t = t_vec(i);
%    integrand = zeros(size(x0));
%    for j = 1:i
%        tau = t_vec(j);
%        integrand = integrand + expm(A * (t - tau)) * B * u_t(:, j) * dt;
%    end
%    x_t(:, i) = expm(A * t) * x0 + integrand;
%end

end
