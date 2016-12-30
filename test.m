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
end

for i = 1:10
    digit = randi(10) - 1;
    disp(['Expected digit ', num2str(digit)])
    k = randi(n_strokes(digit + 1));
    p = digit_classify(strokes_3d{digit + 1}{k});
    disp(['Predicted digit ', num2str(p)])
end
