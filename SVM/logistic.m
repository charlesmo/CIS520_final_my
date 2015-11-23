n = 4000; % train data 
data = [word_train, image_train, train_Y];
data = data( randperm( size(data,1) ),:);

X1train = data(1:n , 1:size(word_train,2)); %4000*5000
X2train = data(1:n , size(word_train,2)+1 : size(word_train,2) + size(image_train,2) ); % 7 * 5000
Ytrain = data(1:n , end); %4000*1

test_X1 = data(n+1:end,1:size(word_train,2)); %1000*5000
test_X2 = data(n+1:end, size(word_train,2)+1 : size(word_train,2) + size(image_train,2) ); 
test_Y = data(n+1:end,end); %1000*1

