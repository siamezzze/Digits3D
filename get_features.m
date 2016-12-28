function [ features ] = get_features( stroke_3d, pca_T, n_anchors)
%GET_FEATURES Transform stroke matrix into vector of features used for
%classification.
% stroke_3d - List of 3D coordinates describing a 3D digit.
% pca_T - Precomputed PCA transform matrix.
% n_anchors - Number of "anchor points" describing a digits.
    
    % 3D to 2D
    stroke_2d = stroke_3d * pca_T;
    % To fixed ammount of points.
    [idx, centers] = kmeans(stroke_2d, n_anchors);
    
    % Restore ordering of anchor points.
    min_point = zeros(1, n_anchors);
    [n, ~] = size(idx);
    for i = n:-1:1
        id = idx(i);
        min_point(id) = i;
    end
    [~, argsort] = sort(min_point);
    anchors = centers(argsort, :);
    
    % Collect features of anchor points.
    % TODO: Try adding more features here.
    
    % Move origin to center of the image;
    maxs = max(anchors);
    mins = min(anchors);
    lengths = maxs - mins;
    center = mins + lengths / 2;
    for i = 1:n_anchors
        anchors(i, :) = anchors(i, :) - center;
    end;
    diag = sqrt(lengths(1) * lengths(1) + lengths(2) * lengths(2));
    maxtheta = pi * 2;
    [theta, rho] = cart2pol(anchors(:, 1), anchors(:, 2));
    rho = rho / diag;
    theta = theta / maxtheta;
    features = zeros(n_anchors * 2, 1);
    for i = 1:n_anchors
        features(i * 2 - 1) = rho(i);
        features(i * 2) = theta(i);
    end
end

