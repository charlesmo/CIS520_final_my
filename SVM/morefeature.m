n = 4200; % train data 
data = [word_train, image_train, train_Y];
data = data( randperm( size(data,1) ),:);

X1train = data(1:n , 1:size(word_train,2)); %4000*5000
X2train = data(1:n , size(word_train,2)+1 : size(word_train,2) + size(image_train,2) ); % 7 * 5000
Ytrain = data(1:n , end); %4000*1

test_X1 = data(n+1:end,1:size(word_train,2)); %1000*5000
test_X2 = data(n+1:end, size(word_train,2)+1 : size(word_train,2) + size(image_train,2) ); 
test_Y = data(n+1:end,end); %1000*1

% lasso regression on original data
lambda_word = 0.011;
w_word = lasso(X1train,Ytrain,'CV',10,'Lambda', lambda_word);
% 
% % First feature - ridge
% K1 = 2000;
% w_word = ridge( Ytrain, X1train, K1, 0 ); % not center and not scale data.  


% add additional feature
Yhat_train = Ytrain - X1train * w_word ; 
K = 1800;
w_image = ridge( Yhat_train, X2train, K, 0); % not center and not scale data.  

test_est_Y = [ones(size(test_X2,1),1),test_X2] * w_image + repmat(test_X1 * w_word,1,3) ;

for i = 1:length(K)
    test_est_Y(test_est_Y(:,i) > mean(test_est_Y(:,i)),i) = 1;
    test_est_Y(test_est_Y(:,i) <= mean(test_est_Y(:,i)),i) = 0;
end

test_Y_mat = repmat(test_Y,1,size(test_est_Y,2));
error = sum(test_est_Y ~= test_Y_mat) / length(test_Y);
acc = 1 - error;