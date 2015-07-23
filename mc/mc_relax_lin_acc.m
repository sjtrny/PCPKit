function [ A, f_vals ] = mc_relax_lin_acc(M, omega, tau, lambda, rho, iterations, tol)
%SOLVE_E_LIN_ACC
%   This function solves the following problem
%   
%   min_A    tau * | A |_* + lambda/2 | P.*E |_F^2 
%   s.t. P.*M = P.*A + P.*E
% 
%   which we convert to the unconstrained problem
% 
%   min_A   tau * | A |_* + lambda/2 | P.*A - P.*M |_F^2
% 
%   solved by accelerated gradient descent
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

alpha = 1;
gamma = 1.1;

f_vals = zeros(iterations, 1);
last_f_value = Inf;

P = zeros(size(M));
P(omega) = 1;

A = zeros(size(M));
Z = zeros(size(M));

for k = 1 : iterations
    
    A_prev = A;
    alpha_prev = alpha;
    
    searching = true;
    while( searching )
        
        partial = lambda * (P.*Z - P.*M);
        V = Z - 1/rho * partial;
        [A, s] = solve_nn(V, tau/rho);
        
        f_vals(k, 1) = tau * sum(s) + lambda/2 * norm(P.*A - P.*M, 'fro')^2;
        
        approx = tau * sum(s) + rho/2 * norm(A - V, 'fro')^2;
        
        if ( f_vals(k, 1) > approx)
            rho = gamma * rho;
        else
            searching = false;
        end
        
    end
    
    alpha = (1 + sqrt(1 + 4*alpha_prev^2)) / 2;
    
    Z = A + ((alpha_prev - 1)/alpha)*(A - A_prev);
    
    if (abs(last_f_value -  f_vals(k, 1)) <= tol)
        break;
    else
        last_f_value = f_vals(k, 1);
    end
    
end

end

