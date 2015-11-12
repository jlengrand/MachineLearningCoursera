function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

tmat = repmat( theta, 1, size(X, 1) );
h_theta = tmat' .* X;

temp_1 = (sum(h_theta,2) - y);
temp_2 = temp_1.^2;

J = 1/(2*m) * sum(temp_2);

% =========================================================================

end
