function [ A, f_vals ] = mc_relax_lin(M, omega, tau, lambda, rho, iterations, tol)
%SOLVE_E_LIN
%   This function solves the following problem
%   
%   min_A    tau * | A |_* + lambda/2 | P.*E |_F^2 
%   s.t. P.*M = P.*A + P.*E
% 
%   which we convert to the unconstrained problem
% 
%   min_A   tau * | A |_* + lambda/2 | P.*A - P.*M |_F^2
% 
%   solved by basic subgradient descent 
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

if ~exist('tau', 'var')
    tau = 10;
end

if ~exist('lambda', 'var')
    lambda = 1;
end

if ~exist('rho', 'var')
    rho = 1;
end

if ~exist('iterations', 'var')
    iterations = 100;
end

if ~exist('tol', 'var')
    tol = 10^-6;
end

f_vals = zeros(iterations, 1);
last_f_value = Inf;

P = zeros(size(M));
P(omega) = 1;

A = zeros(size(M));

for k = 1 : iterations
    
    %% Take a step
    partial = lambda * (P.*A - P.*M);
    
    V = A - 1/rho * partial;
    
    [A, s] = solve_nn(V, tau/rho);
    
    %% Check function value
    f_vals(k, 1) = tau * sum(s) + lambda/2 * norm(P.*A - P.*M, 'fro')^2;
    
    if (abs(last_f_value -  f_vals(k, 1)) <= tol)
        break;
    else
        last_f_value = f_vals(k, 1);
    end
    
end
    
    
end

