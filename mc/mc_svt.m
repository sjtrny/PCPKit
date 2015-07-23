function [ A, f_values, stop_vals ] = mc_svt( M, omega, tau, mu, iterations, tol )
%SOLVE_SVT
%   This function solves the following problem
%   
%   min_A    tau * | A |_*
%   s.t. P.*M = P.*A
%   
%   The method is from
%   "A singular value thresholding algorithm for matrix completion"
%   by Cai, Candes and Shen
%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

if ~exist('M', 'var')
    error('No observation data provided.');
end

if ~exist('omega', 'var')
    error('No observation set provided.');
end

[m, n] = size(M);

if ~exist('mu', 'var')
    mu = 1.2 * (m*n) / nnz(M);
    mu = min(1.9, mu);
end

if ~exist('tau', 'var')
    tau = 5*sqrt(m*n); 
end

if ~exist('iterations', 'var')
    iterations = 500;
end

if ~exist('tol', 'var')
    tol = 10^-4;
end

f_values = zeros(iterations, 1);
stop_vals = zeros(iterations, 1);

Y = M;

P = zeros(size(M));
P(omega) = 1;

for k = 1 : iterations

    %% Step 1, solve for A
    [A, s] = solve_nn(Y, tau);

    %% Step 2, update Y
    Y = Y + mu * (P.*M - P.*A);
        
    %% Get function value
    f_values(k, 1) = tau * sum(s) + .5*norm(A,'fro');
    
    %% Check stopping criteria
    stop_vals(k, 1) = norm(P.*A - P.*M, 'fro') / norm(M, 'fro');
    
    if ( stop_vals(k, 1) <= tol)
        break;
    end
    
end

end

