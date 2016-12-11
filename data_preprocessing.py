import scipy.io
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os.path

strokes = {k: [] for k in range(10)}
train_n = [84, 79, 78, 73, 76, 79, 84, 75, 70, 76]
n = 500
points = []

for digit in range(10):
    for i in range(1, train_n[digit] + 1):
        path = os.path.join("training_data", "stroke_" + str(digit) + "_" + str(i).zfill(3) + ".mat")
        stroke = scipy.io.loadmat(path)['pos']
        strokes[digit].append(stroke)
        points = points + list(stroke)

strokes_2d = {k: [] for k in range(10)}
canvas_2d = {k: [] for k in range(10)}
pca = PCA(n_components=2)
pca.fit(points)
for digit in range(10):
    for i in range(train_n[digit]):
        stroke = strokes[digit][i]
        stroke_2d = pca.transform(stroke)
        strokes_2d[digit].append(stroke_2d)
        mins = np.amin(stroke_2d, axis=0)
        maxs = np.amax(stroke_2d, axis=0)

        lengths = maxs - mins + 2
        lengths = lengths.astype(int)
        canvas = np.zeros(lengths, dtype=np.uint8)

        for j in range(stroke_2d.shape[0]):
            px = stroke_2d[j, :] - mins
            canvas[int(px[0]), int(px[1])] = 255

        canvas_2d[digit].append(canvas)
        plt.imsave(os.path.join("vis", "digit_" + str(digit) + "_" + str(i + 1) + ".png"), canvas, cmap='Greys')
