function [ features ] = get_features( stroke_3d, pca_T, n_anchors)
%GET_FEATURES Transform stroke matrix into vector of features used for
%classification.
% stroke_3d - List of 3D coordinates describing a 3D digit.
% pca_T - Precomputed PCA transform matrix.
% n_anchors - Number of "anchor points" describing a digits.
    
    % 3D to 2D
    stroke_2d = stroke_3d * pca_T;
    
    % Visualisation
    % scatter(stroke_2d(:, 1), stroke_2d(:, 2), 'b.');
    % title('2D digit');
    
    % To fixed ammount of points.
    [idx, centers] = kmeans(stroke_2d, n_anchors);
    
    %figure;
    %scatter(centers(:, 1), centers(:, 2), 'b.');
    %enmr = [1:n_anchors]'; enmr_str = num2str(enmr); 
    %enmr_c = cellstr(enmr_str);
    %d = 0.2;
    %text(centers(:, 1) + d, centers(:, 2) + d, enmr_c);
    %title('Centroids')
    
    % Restore ordering of anchor points.
    min_point = zeros(1, n_anchors);
    [n, ~] = size(idx);
    for i = n:-1:1
        id = idx(i);
        min_point(id) = i;
    end
    [~, argsort] = sort(min_point);
    anchors = centers(argsort, :);
    
    %figure;
    %scatter(anchors(:, 1), anchors(:, 2), 'b.');
    %enmr = [1:n_anchors]'; enmr_str = num2str(enmr); 
    %enmr_c = cellstr(enmr_str);
    %d = 0.2;
    %text(anchors(:, 1) + d, anchors(:, 2) + d, enmr_c);
    %title('Centroids reordered');
    
    % Collect features of anchor points.
    
    % Move origin to center of the image;
    maxs = max(anchors);
    mins = min(anchors);
    lengths = maxs - mins;
    center = mins + lengths / 2;
    diag = sqrt(lengths(1) * lengths(1) + lengths(2) * lengths(2)) / 2;
    [theta, rho] = cart2pol(anchors(:, 1) - center(1), anchors(:, 2) - center(2));
    % Normalize features for better KNN classification.
    maxtheta = pi;
    rho = rho / diag;
    theta = theta / maxtheta;
    features = zeros(1, n_anchors * 2);
    for i = 1:n_anchors
        features(i * 2 - 1) = rho(i);
        features(i * 2) = theta(i);
    end
end

