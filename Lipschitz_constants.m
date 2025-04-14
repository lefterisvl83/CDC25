function [L]=Lipschitz_constants(X)

% Calculate Lipschitz constants for the polyhedra
L = calculateLipschitzConstant(X);

% Output the Lipschitz constants
disp(['Lipschitz constant: ', num2str(L)]);

end


% Function to calculate Lipschitz constant of a polyhedron
function L = calculateLipschitzConstant(P)
    % P is a Polyhedron object
    % A is the matrix of half-space inequalities, representing the polyhedron
    A = P.A;
    
    % Compute the operator norm of A (maximum singular value of A)
    L = norm(A, 'inf');
end

%
end
