# importing basic libraries
import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
from skimage.color import rgb2lab, deltaE_cie76
from collections import Counter
import os

# reading the images
image = cv2.imread('C:/Users/Shubhangi Bhatia/Desktop/nike.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# Defining a function to extract the hex value of colors present in the image
def RGB_HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# In the next function, we are Using Kmeans clusters to arrange the colors in different clusters.
# The algorithm of Kmeans requires a flattened array as input, which is why
# we are required to flatten the NumPy array into an array of 1 row, three columns.
# And using the counter to sort colors RGB value.
def get_colors(image, number_of_colors, show_chart):
    reshaped_image = cv2.resize(image, (600, 400))
    reshaped_image = reshaped_image.reshape(reshaped_image.shape[0]*reshaped_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(reshaped_image)
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB_HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    return rgb_colors


# Finally, we apply our functions to our images to know about the hex value
# of the colours presented in the image
get_colors(image, 5, True)

