function [ label ] = classify_digit(testdata)
%CLASSIFY_DIGIT - classify digit given by set of 3D coordinates.
% pca_T - matrix of PCA transform.
% classifier - classifier model.
% X - 3D stroke data.
    load('model.mat')
    features = get_features(testdata, pca_T, n_anchors)';
    knn_model = fitcknn(train_X, train_Y);
    label = predict(knn_model, features);
end

