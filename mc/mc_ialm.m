function [ A, f_vals, stop_vals ] = mc_ialm(M, omega, tau, mu, iterations, tol )
%SOLVE_IALM
%   This function solves the following problem
%   
%   min_A    tau * | A |_*
%   s.t. M = A + X, P.*X = 0
%   
%   X is not actually noise. Rather it is an auxillary variable
%   to enforce the MC constraint. We expect P.*A = P.*M.
% 
%   The method is from
%   "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matricesn"
%   by Zhouchen Lin, Minming Chen, Yi Ma
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

if ~exist('mu', 'var')
%     mu = 1/norm(M, 2);
    mu = 0.1;
end

if ~exist('tau', 'var')
    tau = 10;
end

if ~exist('iterations', 'var')
    iterations = 100;
end

if ~exist('tol', 'var')
    tol = 10^-6;
end

f_vals = zeros(iterations, 1);
stop_vals = zeros(iterations, 2);

P = zeros(size(M));
P(omega) = 1;

Y = zeros(size(M));
X = zeros(size(M));
R = ones(size(P)) - P;

for k = 1 : iterations

    %% Step 1, solve for A
    
    V = P.*M - X + (1/mu)*Y;
    
    [A, s] = solve_nn(V, tau/mu);
    
    %% Step 2, update E
    old_X = X;
    X = R.*(P.*M - A + (1/mu)*Y);
    
    %% Step 3, update Y
    
    Y = Y + mu*(P.*M - A - X);
    
    %% Get function value
    f_vals(k, 1) = tau * sum(s);
    
    %% Check stopping criteria
    stop_vals(k, 1) = norm(P.*M - A - X, 'fro') / norm(M, 'fro');
    stop_vals(k, 2) = min(mu, sqrt(mu)) * norm(X - old_X, 'fro') / norm(M, 'fro');
    
    if ( stop_vals(k, 1) <= tol && stop_vals(k, 2) <= tol)
        break;
    end
    
end
    
    
end

