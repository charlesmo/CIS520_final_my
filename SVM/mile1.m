%divide data into train and test; total 4998
n = 4000; % train data 
data = [word_train, train_Y];
data = data( randperm( size(data,1) ),:);
train = data(1:n,1:size(word_train,2));
Y = data(1:n,end);
test = data(n+1:end,1:size(word_train,2));
test_Y = data(n+1:end,end);
%divide data into n folders. 
n_folds = 5;  
%do Cross valication to choose parameter lambda 
iterate = 20;
error = zeros(1,20);
lambda = 1:3:30;
errors = zeros(1,length(lambda));
w_initial = rand(size(train,2),1);

for m = 1: length(lambda)
    for i = 1:iterate
        l = lambda(m);
        
        [index] = make_xval_partition( size(train,1), n_folds);
        error(i) = train_error(l, train, Y, index, n_folds,w_initial);
        fprintf('Iterate = %d \n',i);
    end
    errors(m) = sum(error)/iterate;
    fprintf('Lambda = %d \n',m);

end

