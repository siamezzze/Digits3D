import scipy.io
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os.path
import random
VISUALISE = False


def is_corner(p0, p1, p2):
    v1 = p1 - p0
    v1 /= np.linalg.norm(v1)
    v2 = p2 - p1
    v2 /= np.linalg.norm(v2)
    angle = math.acos(np.dot(v1, v2))
    return (angle > math.pi / 2) and (angle < math.pi * 3 / 2)


def number_of_corners(stroke):
    px_0 = stroke[0, :]
    px_1 = stroke[1, :]
    corners = 0
    for j in range(2, len(stroke)):
        px = stroke[j, :]
        step = np.linalg.norm(px - px_1)
        if step < eps:
            continue
        corners += int(is_corner(px, px_1, px_0))
        px_0 = px_1
        px_1 = px
    return corners


# Thanks: http://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


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
        mins = np.amin(stroke_2d, axis=0)
        maxs = np.amax(stroke_2d, axis=0)
        stroke_2d -= mins
        strokes_2d[digit].append(stroke_2d)

        if VISUALISE:
            lengths = maxs - mins + 2
            lengths = lengths.astype(int)
            canvas = np.zeros(lengths, dtype=np.uint8)

            for j in range(stroke_2d.shape[0]):
                px = stroke_2d[j, :]
                canvas[int(px[0]), int(px[1])] = 255

            canvas_2d[digit].append(canvas)
            plt.imsave(os.path.join("vis", "digit_" + str(digit) + "_" + str(i + 1) + ".png"), canvas, cmap='Greys')

eps = 1e-2
features = {k: [] for k in range(10)}
for digit in range(10):
    for i in range(len(strokes_2d[digit])):
        # We have a problem: different amount of points in for different letters.
        # Solve it by clustering down to a fixed amount.
        corners = 0
        stroke_2d = strokes_2d[digit][i]
        kmeans = KMeans(n_clusters=16).fit(stroke_2d)
        stroke_reduced = kmeans.cluster_centers_

        # Now we have fixed amount of points, but we have lost ordering. Let's restore it.
        belongs_to = kmeans.predict(stroke_2d)
        min_point = [0 for _ in range(stroke_reduced.shape[0])]
        for j in range(stroke_2d.shape[0] - 1, -1, -1):
            min_point[belongs_to[j]] = j
        ordering = np.argsort(min_point)
        stroke_reduced = stroke_reduced[ordering]

        maxs = np.amax(stroke_2d, axis=0)
        center = [maxs[0] / 2, maxs[1] / 2]
        diag = np.linalg.norm(center)
        stroke_polar = []
        for j in range(stroke_2d.shape[0]):
            px = stroke_2d[j, :]
            rho, theta = cart2pol(px[0] - center[0], px[1] - center[1])
            rho /= diag
            stroke_polar += [rho, theta]
        features[digit].append(stroke_polar)

        # Visualisation
        if VISUALISE:
            mins = np.amin(stroke_2d, axis=0)
            maxs = np.amax(stroke_2d, axis=0)

            lengths = maxs - mins + 2
            lengths = lengths.astype(int)
            canvas = np.zeros(lengths, dtype=np.uint8)

            for j in range(stroke_reduced.shape[0]):
                px = stroke_reduced[j, :]
                canvas[int(px[0]), int(px[1])] = 255 - 5 * j

            plt.imsave(os.path.join("kmeans", "digit_" + str(digit) + "_" + str(i + 1) + ".png"), canvas, cmap='Greys')