function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% forward propagation

X = [ones(m, 1) X];
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% [val, p] = max(a3, [], 2);

% reshaping y to have vectors
y2 = zeros(size(y, 1), num_labels);
for ii = 1:size(y2, 1)
    y2(ii, y(ii)) = 1;
end

% Cost without regularization
for i = 1:m
    temp1 = - y2(i, :) * log(a3(i, :))';
    temp2 = (1-y2(i, :))*(log(1- a3(i, :)))';
    temp3 = temp1 - temp2;
    J = J + temp3;
end
J = J / m;

% Adding regularization
Theta11 = Theta1;% Theta1 without bias;
Theta11(:, 1) = [];
Theta22 = Theta2;
Theta22(:, 1) = [];

Theta11s = Theta11.^2;
Theta22s = Theta22.^2;

reg = (lambda/(2*m))*(sum(Theta11s(:)) + sum(Theta22s(:)));
J = J + reg;

% -------------------------------------------------------------

% Backpropagation
% Here we are going to calculate Theta1_grad and Theta2_grad 

for i = 1:m
    % step 1
    a1 = X(i, :);
    a2t = a2(i, :);
    z2t = [1 z2(i, :)];
    a3t = a3(i, :);
    z3t = z3(i, :);
    y_t = y2(i, :);
    
    % step 2
    delta3 = a3t - y_t;
    
    % step 3
    t2 = (Theta2' * delta3');
    delta2 = t2'.*(sigmoidGradient(z2t));
    
    % step 4
    delta2(:, 1) = [];
    Theta1_grad = Theta1_grad + (delta2' * a1);
    Theta2_grad = Theta2_grad + (delta3' * a2t); 
end

% step 5
Theta1_grad = (1/m).*Theta1_grad;
Theta2_grad = (1/m).*Theta2_grad;

% Adding regularization
temp_reg1 = Theta1;
temp_reg1(:, 1) = zeros(size(temp_reg1(:, 1)));
% temp_reg1(1:end, 2:end) = (lambda/m) * Theta1(1:end, 2:end);
temp_reg2 = Theta2;
temp_reg2(:, 1) = zeros(size(temp_reg2(:, 1)));
% temp_reg2(1:end, 2:end) = (lambda/m) * temp_reg2(1:end, 2:end);
Theta1_grad = Theta1_grad + (lambda/m) * temp_reg1;
Theta2_grad = Theta2_grad + (lambda/m) * temp_reg2;

% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
