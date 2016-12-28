function [ label ] = knn_classify( train_X, train_Y, X, k, n_classes )
%KNN_CLASSIFY - Predict the class of a sample using k nearest neighbours.
% train_X - train samples (each row corresponds to a single observation).
% train_Y - ground truth answers for train samples.
% X - data sample to classify (row).
% k - number of neighbours used.
% n_classes - number of classes.
    [n, ~] = size(train_X);
    distances = zeros(1, n);
    for i = 1:n
        distances(i) = norm(train_X(i, :) - X);
    end
    [~, idx] = sort(distances);
    neighbours_Y = train_Y(idx);
    votes = zeros(n_classes, 1);
    for i = 1:k
        y = neighbours_Y(i) + 1;
        votes(y) = votes(y) + 1;
    end
    [~, y_best] = max(votes);
    label = y_best(1) - 1;
end

