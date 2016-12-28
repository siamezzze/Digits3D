function [ label ] = classify_digit( pca_T, classifier, X)
%CLASSIFY_DIGIT - classify digit given by set of 3D coordinates.
% pca_T - matrix of PCA transform.
% classifier - classifier model.
% X - 3D stroke data.
    n_anchors = 16;
    features = get_features(X, pca_T, n_anchors)';
    label = predict(classifier, features);
end

