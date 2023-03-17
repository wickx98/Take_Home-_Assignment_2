import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(gray)

seeds = [(255, 255)]

threshold = 10

# mark every seed point in mask as 1
for seed in seeds:
    mask[seed[0], seed[1]] = 1

# 4 connected neighbors
neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

for seed in seeds:
    for neighbor in neighbors:
        coord = {
            'x': seed[0] + neighbor[0],
            'y': seed[1] + neighbor[1]
        }

        # check if the neighbor is out of bounds
        if coord['x'] < 0 or coord['x'] >= gray.shape[0] or coord['y'] < 0 or coord['y'] >= gray.shape[1]:
            continue

        # check if the neighbor is already marked
        if mask[coord['x'], coord['y']] != 0:
            continue

        # check if the neighbor is within the threshold
        if gray[coord['x'], coord['y']] >= (gray[seed[0], seed[1]] - threshold) and gray[coord['x'], coord['y']] <= (gray[seed[0], seed[1]] + threshold):
            mask[coord['x'], coord['y']] = 1
            seeds.append((coord['x'], coord['y']))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray, cmap='gray')
ax[1].imshow(mask, cmap='gray')
plt.show()