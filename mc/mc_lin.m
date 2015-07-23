function [ A, f_vals ] = mc_lin(M, omega, tau, mu, rho, iterations, tol)
%SOLVE_LIN
%   This function solves the following problem
%   
%   min_A    tau * | A |_* 
%   s.t. P.*M = P.*A + P.*E
% 
%   which we convert to the unconstrained problem
% 
%   min_A   tau * | A |_* + < Y, P.*A - P.*M > + mu/2 | P.*A - P.*M |_F^2
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

if ~exist('mu', 'var')
    mu = 1;
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
Y = zeros(size(M));

for k = 1 : iterations
    
    %% Take a step
    partial = mu * (P.*A - (P.*M - 1/mu * Y));
    
    V = A - 1/rho * partial;
    
    [A, s] = solve_nn(V, tau/rho);
    
    %% Step 2, update Y
    Y = Y + mu * (P.*A - P.*M);
    
    %% Check function value
    f_vals(k, 1) = tau * sum(s);
    
    if (abs(last_f_value -  f_vals(k, 1)) <= tol)
        break;
    else
        last_f_value = f_vals(k, 1);
    end
    
end
    
    
end

