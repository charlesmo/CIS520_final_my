function [test_est_Y1] = prebag(X1train,X2train,Ytrain,test_X1,test_X2,lambda,K)


X1 = X1train;
X2 = X2train;
Y = Ytrain;

%PCA version
[coff , X_pca] = princomp(X1);
test_pca = ( test_X1 - repmat( mean(test_X1),size(test_X1,1),1 ) ) * coff( :,1:600);
% Lasso regression
w1 = lasso(X_pca(:,1:600), Y,'Lambda', lambda);

% ridge regression
Yhat_train = Y - X_pca(:,1:600) * w1 - mean( X_pca(:,1:600) * w1 ) ; 
w2 = ridge( Yhat_train, X2, K, 0); % not center and not scale data.  

% Cal error
test_est_Y1 = [ones(size(test_X2,1),1),test_X2] * w2 + repmat(test_pca * w1,1,length(K)) ;
test_est_Y1(test_est_Y1>mean(test_est_Y1)) = 1;
test_est_Y1(test_est_Y1<=mean(test_est_Y1)) = 0;

%test_Y_mat = repmat(test_Y,1,size(test_est_Y,2));
% 
% k = @(x,x2) kernel_intersection(x,x2);
% [info,test_est_Y2] = kernel_libsvm(X1train, Ytrain, test_X1, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE

