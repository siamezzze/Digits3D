n_strokes = [84 79 78 73 76 79 84 75 70 76];
strokes_3d = cell(10, 1);
strokes_digit = cell(10, 1);

% 0. Read the data.
for digit = 0:9
    kmax = n_strokes(digit + 1);
    strokes_3d{digit + 1} = cell(kmax, 1);
    for k = 1:kmax
        fname = sprintf('training_data/stroke_%i_%03i.mat', digit, k);
        stroke = load(fname);
        strokes_3d{digit + 1}{k} = stroke.pos;
    end
    strokes_digit{digit + 1} = cat(1, strokes_3d{digit + 1}{:});
end

% Combine all points to form a base for PCA.
all_points = cat(1, strokes_digit{:});

all_points = all_points';
[coef, score, latent] = pca(all_points);
pca_T = score(:, 1:2);
% matrix of PCA transform.

n_anchors = 16;
features_matrix = zeros(sum(n_strokes), 2 * n_anchors);
answers = zeros(sum(n_strokes), 1);
i = 1;
for digit = 0:9
    kmax = n_strokes(digit + 1);
    for k = 1:kmax
        stroke = strokes_3d{digit + 1}{k};
        features_row = get_features(stroke, pca_T, n_anchors);
        features_matrix(i, :) = features_row;
        answers(i, :) = digit;
        i = i + 1;
    end
end

train_X = features_matrix;
train_Y = answers;
K = 5;
save('model.mat', 'pca_T', 'train_X', 'train_Y', 'K', 'n_anchors');