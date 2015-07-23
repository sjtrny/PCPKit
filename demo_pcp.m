%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

paths = ['common:', 'pcp:'];
addpath(paths);

rng(1);

rows = 100;
n_space = 5;
cluster_size = 50;

A = rand(rows, n_space) * rand(n_space, n_space);

permute_inds = reshape(repmat(1:n_space, cluster_size, 1), 1, n_space * cluster_size );
A = A(:, permute_inds);

corruption = 0;

N = randn(size(A)) * corruption;

X = A + N;

A_est = pcp_fro(X, 10);

rmpath(paths);