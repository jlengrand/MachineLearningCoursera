function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Calculating J
theta_t_x = X*theta;
h_theta = sigmoid(theta_t_x);

part_1 = -y'*log(h_theta);
part_2 = (1 - y)' * log(1 - h_theta);
J = 1 / m * sum(part_1 - part_2);

% Calculating g
temp_1 = sigmoid(X*theta) - y;
temp_2 = repmat(temp_1, 1, size(X, 2));
grad = 1/m * sum(X .* temp_2);

% =============================================================

end
