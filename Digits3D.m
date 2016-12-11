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

% 1. Data preprocessing (convert to 2d).
% Combine all points to form a base for PCA.
all_points = cat(1, strokes_digit{:});

all_points = all_points';
[coef, score, latent] = pca(all_points);
pca_T = score(:, 1:2);
% matrix of PCA transform.

strokes_2d = cell(10, 1);
for digit = 0:9
    kmax = n_strokes(digit + 1);
    strokes_2d{digit + 1} = cell(kmax, 1);
    for k = 1:kmax
        stroke = strokes_3d{digit + 1}{k};
        stroke_2d = stroke * pca_T;
        strokes_2d{digit + 1}{k} = stroke_2d;
    end
end

% Example:
stroke_2d = strokes_2d{3}{15};
scatter(stroke_2d(:, 1), stroke_2d(:, 2), 'b.');

% 3. Feature selection.
eps = 0.001;
angles = cell(10, 1);
angle_hist = cell(10, 1);
for digit = 0:9
    kmax = n_strokes(digit + 1);
    angles{digit + 1} = cell(kmax, 1);
    angle_hist{digit + 1} = cell(kmax, 1);
    for k = 1:kmax
        stroke = strokes_2d{digit + 1}{k};
        imax = size(stroke, 1);
        
        angles{digit + 1}{k} = zeros(imax - 2, 1);
        angle_hist{digit + 1}{k} = zeros(13, 1);
        for i = 3:imax
            vector_a_start = stroke(i - 2, :);
            vector_a_finish = stroke(i - 1, :);
            vector_a = vector_a_finish - vector_a_start;
            vector_b_start = stroke(i - 1, :);
            vector_b_finish = stroke(i, :);
            vector_b = vector_b_finish - vector_b_start;
            
            % Points are too close, angle measurement is unstable.
            if norm(vector_a) < eps || norm(vector_b) < eps
                continue
            end
            
            % Normalize (cast to unit vectors).
            vector_a = vector_a / norm(vector_a);
            vector_b = vector_b / norm(vector_b);
            
            x1 = vector_a(1); y1 = vector_a(2);
            x2 = vector_b(1); y2 = vector_b(2);
            
            angle = round(mod(atan2(y2 - y1, x2 - x1), 2 * pi) * 180 / pi);
            angle_d = fix(angle / 30);
            angles{digit + 1}{k}(i - 2, 1) = angle_d * 30;
            
            angle_hist{digit + 1}{k}(angle_d + 1, 1) = ...
                angle_hist{digit + 1}{k}(angle_d + 1, 1) + 1;
        end
    end
end

disp(angles{3}{15});