load('model.mat')
conf_matrix = zeros(10);
[n, ~] = size(train_X);
test_part = 70;

n_validations = 10;
K = 3;
for i = 1:n_validations
    val_idxs = randsample(n, test_part);
    train_idxs = setdiff(1:n, val_idxs);
    
    train_X_part = train_X(train_idxs, :);
    train_Y_part = train_Y(train_idxs, :);
    
    val_X_part = train_X(val_idxs, :);
    val_Y_part = train_Y(val_idxs, :);
    
    n_val = test_part;
    val_Y_predicted = zeros(n_val, 1);
    
    for j = 1:n_val
        x = val_X_part(j, :);
        label = knn_classify(train_X_part, train_Y_part, x, K, 10);
        val_Y_predicted(j) = label;
        y_pred = val_Y_part(j);
        conf_matrix(y_pred + 1, label + 1) = conf_matrix(y_pred + 1, label + 1) + 1;
    end
    
    accuracy = double(sum(val_Y_part == val_Y_predicted)) / n_val;
    fprintf('%i. Accuracy = %f\n', i, accuracy);
end
disp('Confusion matrix:');
disp(conf_matrix);