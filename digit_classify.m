function [ label ] = digit_classify(testdata)
%DIGIT_CLASSIFY - classify digit given by set of 3D coordinates.
% testdata - N x 3 matrix of 3D coordinates describing digit.
    load('model.mat')
    features = get_features(testdata, pca_T, n_anchors);
    label = knn_classify(train_X, train_Y, features, K, 10);
end

