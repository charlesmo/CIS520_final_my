function [info,yhat] = kernel_libsvm(X, Y, Xtest, kernel)
% Trains a SVM using libsvm and evaluates on test data.
%
% Usage:
%
%   [TEST_ERR INFO] = KERNEL_LIBSVM(X, Y, XTEST, YTEST, KERNEL)
%
% Runs training and testing of a SVM with the given kernel function, using
% cross validation to choose regularization parameter C. X, Y, XTEST, and
% YTEST should be created using MAKE_SPARSE. KERNEL is a FUNCTION HANDLE to
% the appropriate KERNEL function, which must take ONLY TWO PARAMETERS
% K(X,X2).
%
% EXAMPLES:
%
% Compute error using a poly kernel with P=2:
%
% >> k = @(x,x2) kernel_poly(x, x2, 1);
% >> [test_err info] = kernel_libsvm(X, Y, Xtest, Ytest, k)
%
% The first step is necessary to create a function that only depends on two
% arguments from the KERNEL_POLY function which takes 3.

% Compute kernel matrices for training and testing.
K = kernel(X, X);
Ktest = kernel(X, Xtest);

crange = 0.0007;

% Use built-in libsvm cross validation to choose the C regularization
% parameter.

% crange = [0.00005,0.0001,0.00015,0.0002];
% for i = 1:numel(crange)
%     acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
% end
% [~, bestc] = max(acc);
% fprintf('Cross-val chose best C = %g\n', crange(bestc));

Ytest = randi([0 1], size(Ktest,1), 1);
% Train and evaluate SVM classifier using libsvm
model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange));

[yhat, acc, vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);

% Optionally we can look at more information from training/testing.
info.vals = vals;
info.yhat = yhat;
info.model = model;




    
    