import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import chain
from PIL import Image
import glob

import re

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_scaled_letter(unicode_num=65, fontname="Arial", target_size=20, dpi=20, show = False):
    """Plots a letter and scales its bounding box to exactly match target_size×target_size pixels."""
    print(f"output/{fontname}_{unicode_num}")
    letter = chr(unicode_num)

    fig, ax = plt.subplots(figsize=(target_size/dpi, target_size/dpi), dpi=dpi)
    fontsize = 100  # Initial guess

    ax.clear()
    text = ax.text(0.5, 0.5, letter, fontsize=fontsize, fontname=fontname,
                    ha='center', va='center')

    plt.draw()  # Render the figure
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox_height = bbox.height  # Bounding box height in pixels


    fontsize *= (target_size / bbox_height)
    #print(fontsize)

    # Final render with correct font size
    ax.clear()
    text = ax.text(0.5, 0.5, letter, fontsize=fontsize, fontname=fontname,
                   ha='center', va='center')

    plt.draw()
    

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Save final image
    plt.savefig(f"output/{fontname}_{unicode_num}.png", dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)  # Explicitly close the figure to prevent memory leaks


def generator(font_list):
    # A = 65
    # a = 97

    letters_range = list(chain(range(65, 65+26), range(97, 97+26)))

    for f in font_list:
        for n in letters_range:
            plot_scaled_letter(unicode_num=n, fontname=f)

def char_images_reader(output = 'table.csv'):
    font_names = []
    char_numbers = []
    char_images = []

    pattern = r"output\\([\w\s]+)_(\d+)\.png"

    for im_path in glob.glob("output/*.png"):
        match = re.search(pattern, im_path)

        if match:
            font_name = match.group(1)
            font_names.append(font_name)

            char_number = int(match.group(2))
            char_numbers.append(char_number)


            # Open image and process
            image = Image.open(im_path).convert("L")
            arr = 255 - np.asarray(image)

            char_images.append(arr.flatten())
    
    df = pd.DataFrame(char_images)

    df.insert(0, 'font', font_names)
    df.insert(1, 'unicode', char_numbers)

    df.to_csv(output, index=False)

    '''font_names = np.asarray(font_names)
    char_numbers = np.asarray(char_numbers)
    char_images = np.asarray(char_images)'''

def plot_variances(pca, n_components = 25):
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components+1), pca.explained_variance_ratio_[:n_components], color='b', alpha=0.7)

    # Add labels and title
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance per Principal Component')
    plt.xticks(range(1, n_components + 1))  # Ensure x-ticks match component numbers
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig("explained_variance.png", dpi=300, bbox_inches='tight') 
    # Show plot
    plt.show()


fonts = ["Arial", "Times New Roman", "Courier New", "Comic Sans MS", "DejaVu Sans"]

# generate the font picture folder
#generator(fonts)

# reade the folder and generate a csv
#char_images_reader()

df = pd.read_csv('table.csv')

# font and unicode number
labels = df.iloc[:, :2]
features = df.iloc[:, 2:]

print(features.head)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.95)  # Retain 95% variance
principal_components = pca.fit_transform(features_scaled)


pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Concatenate labels with PCA result
final_df = pd.concat([labels, pca_df], axis=1)

plot_variances(pca)

