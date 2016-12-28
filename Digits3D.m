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

%{
% angle histograms
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
%}

eps = 0.001;
features =  cell(10, 1);
n_clusters = 16;

for digit = 1:10
    features{digit} = cell(length(strokes_2d{digit}),1);
    for i = 1:length(strokes_2d{digit}) 
        % We have a problem: different amount of points in for different letters.
        % Solve it by clustering down to a fixed amount.
        corners = 0;
        stroke_2d = strokes_2d{digit}{i};
        [belongs_to, stroke_reduced] = kmeans(stroke_2d,n_clusters);
        % Now we have fixed amount of points, but we have lost ordering. Let's restore it.
        min_point = 1:size(stroke_reduced,1);
        for j = size(stroke_2d,1):-1:1
            min_point(belongs_to(j)) = j;
        end
        [ordered_arr,ordering] = sort(min_point);
        stroke_reduced1 = stroke_reduced(ordering,:);

        maxs = max(stroke_2d);
        center = [maxs(1) / 2, maxs(2) / 2];
        diag = norm(center);
        stroke_polar = [];
        for j = 1: size(stroke_reduced,1)
            px = stroke_reduced(j, :);
            [theta,rho] = cart2pol(px(1) - center(1), px(2) - center(2));
            rho = rho/diag;
            stroke_polar = [stroke_polar ; [rho, theta]];
        end
        features{digit}{i} = stroke_polar;

        % Visualisation
       %{ 
            mins = np.amin(stroke_2d, axis=0)
            maxs = np.amax(stroke_2d, axis=0)

            lengths = maxs - mins + 2
            lengths = lengths.astype(int)
            canvas = np.zeros(lengths, dtype=np.uint8)

            for j in range(stroke_reduced.shape[0]):
                px = stroke_reduced[j, :]
                canvas[int(px[0]), int(px[1])] = 255 - 5 * j

            plt.imsave(os.path.join("kmeans", "digit_" + str(digit) + "_" + str(i + 1) + ".png"), canvas, cmap='Greys')
        %}
    end
end

%{
k_test = 10
test_subset = {k: random.sample(range(train_n[k]), k_test) for k in range(10)}

train_X = []
train_Y = []
test_X = []
test_Y = []
for digit in range(10):
    for j in range(len(features[digit])):
        feature = features[digit][j]
        if j in test_subset[digit]:
            test_X.append(feature)
            test_Y.append(digit)
        else:
            train_X.append(feature)
            train_Y.append(digit)

train_X = np.array(train_X, dtype=np.float)
train_Y = np.array(train_Y, dtype=int)
neighbours = KNeighborsClassifier(n_neighbors=5)
neighbours.fit(train_X, train_Y)
classified = neighbours.predict(test_X)

cm = confusion_matrix(test_Y, classified)
plt.imshow(cm, interpolation='nearest')
plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel("Real digits")
plt.ylabel("Predicted digits")
plt.title("Confusion matrix")
plt.colorbar()
plt.show()
%}