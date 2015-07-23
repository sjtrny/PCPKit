function [ A, E ] = sel_pcp( M, tau, lambda, exact_ind, approx_ind )
%SOLVE_REG_LIN_ALM
%   This function solves the following problem
%   
%   min_{A, E}   | A |_* + lambda/2 | E |_F^2
%   s.t. P.*M = P.*A
%   s.t. N.*M = N.*A + N.*E
%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

max_iterations = 100;
func_vals = zeros(max_iterations,1);

tol_1 = 1*10^-2;
tol_2 = 1*10^-4;

eta_A = 1;
eta_E = 1;

mu = 2;
mu_max = 10;

gamma_0 = 1.1;

Y_1 = zeros(size(M));
Y_2 = zeros(size(M));

P = zeros(size(M));
P(exact_ind) = 1;

N = zeros(size(M));
N(approx_ind) = 1;

% A = zeros(size(M));
A = P.*M;
E = zeros(size(M));

normPM = norm(P.*M, 'fro');
normNM = norm(N.*M, 'fro');

for k = 1 : max_iterations

    prev_A = A;
    prev_E = E;
    
    % Update A
    partial_A = mu * (P.*A - (P.*M - 1/mu * Y_1)) + mu * (N.*A - (N.*M - N.*E - 1/mu * Y_2));
    
    [A, s] = solve_nn(prev_A - (1/(mu*eta_A)) * partial_A, tau/(mu*eta_A));
    
    % Update E
    partial_E = mu * (N.*E - (N.*M - N.*A - 1/mu * Y_2));
    
    E = solve_l2(prev_E - (1/(mu*eta_E)) * partial_E, lambda/(mu*eta_E));

    % Update Y_1 and Y_2
    
    Y_1 = Y_1 + mu * P.*(A - M);
    Y_2 = Y_2 + mu * N.*(A + E - M);
    
    q = mu/norm(M, 'fro') * max([sqrt(eta_A)*norm(A - prev_A,'fro'), sqrt(eta_E)*norm(E - prev_E,'fro')]);
    
    % Update mu
    if q < tol_2
        gamma = gamma_0;
    else
        gamma = 1;
    end
    
    mu = min(mu_max, gamma * mu);  
    
    func_vals(k) = sum(s) + (lambda/2) * norm(E, 'fro')^2;
    
    % Check convergence
    if ( norm(P.*(A - M), 'fro')/normPM < tol_1 && norm(N.* (A + E - M),'fro')/normNM < tol_1 && q < tol_2 )
        break;
    end
    
end

end