paths = [genpath('common'), genpath('mc')];
addpath(paths);

rng(1);

n = 150;
n_blocks = 3;

A = get_block_diag(n, n_blocks);

df = n_blocks*(n + n - n_blocks);

oversampling = 5;
m = min(oversampling*df,round(.99*n*n));

omega = randsample(n*n, m);

M = zeros(size(A));
M(omega) = A(omega);

A_est = mc_lin(M, omega);

rmpath(paths);