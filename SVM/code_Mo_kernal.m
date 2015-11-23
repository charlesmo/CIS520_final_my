%% CIS520 final_project group:Terminator

%% Method 2: SVM + Kernal
Initialize

y_1 = gender_train_new;
y_2 = gender_test_new;
x_1_words = words_train_new;
x_2_words = words_test_new;

observation_num = size(x_1_words,1);
feature_num = size(x_1_words,2);



%PCA the feature selection part
num_pca = 700;
[pc,score] = pca(x_1_words);
loadings_pca = pc(:,1:num_pca) ; % 5000 * num_pca


%transform data to the PCA space
x_1_pca = x_1_words * loadings_pca;
x_2_pca = x_2_words * loadings_pca;



%Prediction
k = @(x,x2) kernel_intersection(x,x2);

[info,est_Y] = kernel_libsvm(x_1_words, y_1, x_2_words, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE





