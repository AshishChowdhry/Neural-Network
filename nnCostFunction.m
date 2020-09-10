function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X=[ones(m,1) X];
J=0;
Thet1=Theta1;
Thet1(:,1)=0;
Thet2=Theta2;
Thet2(:,1)=0;

z2=X*Theta1';                            
a2=sigmoid(X*Theta1');
n=size(a2,1);
a2 = [ones(n, 1) a2];
a3 = sigmoid(a2*Theta2');

for i=1:m
    out=zeros(num_labels,1);
    out(y(i))=1;
    b=(a3(i,:))';
    J=J+((1/m)*((-out'*log(b)-(1-out)'*log(1-b))));
end;
J=J+(lambda/(2*m))*(sum(sum(Thet1.^2))+sum(sum(Thet2.^2)));
    
delta1=zeros(size(Theta1));
delta2=zeros(size(Theta2));

for i=1:m
    out=zeros(num_labels,1);
    out(y(i))=1;
    sd3=(a3(i,:))'-out;
    sd2=Theta2'*sd3;
    sd2=sd2(2:end);
    sd2=sd2.*(sigmoidGradient(z2(i,:)))';
    delta1=delta1+sd2*X(i,:);
    delta2=delta2+sd3*a2(i,:);
end;
Theta1_grad=delta1/m+(lambda/m)*Thet1;
Theta2_grad=delta2/m+(lambda/m)*Thet2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
