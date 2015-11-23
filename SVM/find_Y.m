function [w_now,test_Y] = find_Y(X ,Y,test_X , lambda,w_prev);

fun = @(w)(Y-X*w)'*(Y-X*w) + lambda *  sum(abs(w));
[w_now,~] = fminsearch(fun,w_prev);

test_Y = test_X * w_now;

test_Y(test_Y<0.5,:)= 0;
test_Y(test_Y>=0.5,:) = 1;

%wmle = inv(X'*X)*X'*Y;
%ratio = sum(power(w,2))/ sum(power(wmle,2))
