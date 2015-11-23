function [error] = train_error(lambda, X, Y, index,num_folds,w_initial)

% Prepare:
errors = zeros(1,num_folds);
w_prev = w_initial;
w_now = zeros(size(X,2),1);

for i = 1:num_folds
   % [~,N] = find(index == i); % J store the postion of fold
    trainPoints = X;
    trainPoints(index==i,:) = [];   % get ride of the ith data
    trainLabels = Y;
    trainLabels(index==i) = [];
    testPoints = X(index==i,:);      %Si
    testErrorLabels = Y(index==i); % Si
    [w_now,estimateLabel] = find_Y(trainPoints,trainLabels,testPoints,lambda,w_prev);
    errors(i)= sum( testErrorLabels ~= estimateLabel);
    errors(i)= errors(i)/size(estimateLabel,1);
    w_prev = w_now;
    fprintf('num_folds = %d \n',i);

end
error = sum(errors)/num_folds;
    