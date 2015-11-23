%% Initialize
load('data_train/data_train.mat');
total_num = size(words_train,1);
total_feature_num = size(words_train,2);

testing_num = 700;

testing_index = sort(randperm(total_num,testing_num));
total_index = 1:total_num;
training_index = setxor(total_index,testing_index);

words_train_new = words_train(training_index,:);
words_test_new = words_train(testing_index,:);

gender_train_new = gender_train(training_index,:);
gender_test_new = gender_train(testing_index,:);

image_features_train_new = image_features_train(training_index,2:end);
image_features_test_new = image_features_train(testing_index,2:end);

