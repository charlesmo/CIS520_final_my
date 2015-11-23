lambda = 0.005; % 0.011  
K = 1800;

%PCA version
[coff , X_pca] = princomp(word_train);
test_pca =  test_word * coff(:,1:600);
% Lasso regression
w1 = lasso(X_pca(:,1:600), train_Y,'Lambda', lambda);

% ridge regression
Yhat_train = train_Y - X_pca(:,1:600) * w1 - mean( X_pca(:,1:600) * w1 ) ; 
w2 = ridge( Yhat_train, image_train, K, 0); % not center and not scale data.  

% Cal error
test_est_Y = [ones(size(image_test,1),1),image_test] * w2 + repmat(test_pca * w1,1,length(K)) ;
test_est_Y(test_est_Y>mean(test_est_Y)) = 1;
test_est_Y(test_est_Y<=mean(test_est_Y)) = 0;