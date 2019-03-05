function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

a=sigmoid(theta'*X');
a=a';
b=-y.*log(a);
c=1-y;
aa=1-a;
bb=-c.*log(aa);

J1=sum(b+bb)/m;

thetaa=theta;
thetaa(1)=0;
J2=sum(thetaa.*thetaa);
J2=J2*lambda/(2*m);

J=J1+J2;


grad=X'*(a-y);
grad=grad./m;
gradd=(lambda.*theta)./m;
grad=grad+gradd;
grad(1,1)=grad(1,1)-lambda*theta(1)/m;
% =============================================================

end
