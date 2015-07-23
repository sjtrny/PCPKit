%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

paths = [genpath('common'), genpath('selective_pcp')];
addpath(paths);

rng(1);

n = 150;
n_blocks = 3;
n_empty_cols = 50;

A = get_block_diag(n, n_blocks);

empty_cols = randsample(size(A, 2), n_empty_cols);
ind_mat = reshape((1: (n*n)), n, n);

df = n_blocks*(n + n - n_blocks);
oversampling = 5;
m = min(oversampling*df,round(.99*n*n));
omega_rand = randsample(n*n, m);
omega_cols = reshape(ind_mat(:, setdiff(1:n, empty_cols)), n*(n-n_empty_cols), 1);
omega = intersect(omega_cols, omega_rand);

psi = reshape(ind_mat(:, empty_cols), n*n_empty_cols, 1);

X = A + 0.1 * randn(size(A));

M = zeros(n);
M(omega) = A(omega);
M(psi) = X(psi);

A_est = sel_pcp(M, 10, 7.5, omega, psi);

rmpath(paths);