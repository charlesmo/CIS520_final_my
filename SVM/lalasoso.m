n = 4500; % train data 
data = [word_train, image_train, train_Y];
data = data( randperm( size(data,1) ),:);

X1train = data(1:n , 1:size(word_train,2)); %4000*5000
X2train = data(1:n , size(word_train,2)+1 : size(word_train,2) + size(image_train,2) ); % 7 * 5000
Ytrain = data(1:n , end); %4000*1

test_X1 = data(n+1:end,1:size(word_train,2)); %1000*5000
test_X2 = data(n+1:end, size(word_train,2)+1 : size(word_train,2) + size(image_train,2) ); 
test_Y = data(n+1:end,end); %1000*1


lambda = 0.005; % 0.011  
K = 1800;
[Y1] = prebag(X1train,X2train,Ytrain,test_X1,test_X2,lambda,K);
[Y2] = prebag(X1train,X2train,Ytrain,test_X1,test_X2,lambda,K);
[Y3] = prebag(X1train,X2train,Ytrain,test_X1,test_X2,lambda,K);


Y = ((Y1 + Y2 + Y3)>1.5) ;
error = mean(Y ~= test_Y ) ;
%PCA version
% [coff , X_pca] = princomp(X1train);
% test_pca = ( test_X1 - repmat( mean(test_X1),size(test_X1,1),1 ) ) * coff( :,1:750);
% % Lasso regression
% w1 = lasso(X_pca(:,1:750), Ytrain,'Lambda', lambda);
% 
% % ridge regression
% Yhat_train = Ytrain - X_pca(:,1:750) * w1 - mean( X_pca(:,1:750) * w1 ) ; 
% w2 = ridge( Yhat_train, X2train, K, 0); % not center and not scale data.  
% 
% % Cal error
% test_est_Y1 = [ones(size(test_X2,1),1),test_X2] * w2 + repmat(test_pca * w1,1,length(K)) ;
% test_est_Y1(test_est_Y1>mean(test_est_Y1)) = 1;
% test_est_Y1(test_est_Y1<=mean(test_est_Y1)) = 0;
% 
% %test_Y_mat = repmat(test_Y,1,size(test_est_Y,2));
% % 
% % k = @(x,x2) kernel_intersection(x,x2);
% % [info,test_est_Y2] = kernel_libsvm(X1train, Ytrain, test_X1, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE
% 
% 
% error = mean(test_est_Y1 ~= test_Y ) ;
% acc = 1 - error;

% lambda = 0.01;    
% w = lasso(Xtrain,Ytrain,'CV',10,'Lambda', lambda);
% test_est_Y = test_X * w;
% 
% for i = 1:length(lambda)
%     test_est_Y(test_est_Y(:,i)>mean(test_est_Y(:,i)),i) = 1;
%     test_est_Y(test_est_Y(:,i)<=mean(test_est_Y(:,i)),i) = 0;
% end
% 
% test_Y_mat = repmat(test_Y,1,size(test_est_Y,2));
% error = sum(test_est_Y ~= test_Y_mat) / length(test_Y);
% acc = 1 - error

% w=  = lasso(word_train, train_Y, 'CV',10,'Lambda',0.01);
% test_est_Y 

