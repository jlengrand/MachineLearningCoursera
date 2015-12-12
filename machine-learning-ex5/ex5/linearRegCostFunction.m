function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


theta_temp = theta;
theta_temp(1) = 0;

% h_theta = sigmoid(X*theta);
h_theta = (X*theta);
temp = (h_theta - y).^2;

temp1 = (1/(2 * m)) * sum(temp);
temp2 = (lambda/(2 * m)) * sum(theta_temp.^2);
J = temp1 + temp2;

% =========================================================================

t11 = sigmoid(X*theta) - y;
t12 = repmat(t11, 1, size(X, 2));
temp11 = (1/m) * sum(X .* t12);

temp22 = (lambda / m) * theta_temp;
grad = temp11' + temp22;


% =========================================================================

grad = grad(:);

end
