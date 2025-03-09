import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import chain
import glob
import re

from PIL import Image

def plot_scaled_letter(unicode_num=65, fontname="Arial", target_size=20, dpi=30, show = False):
    #print(f"output/{fontname}_{unicode_num}")
    letter = chr(unicode_num)
    try:
        fig, ax = plt.subplots(figsize=(target_size/dpi, target_size/dpi), dpi=dpi)
        fontsize = 100  # Initial guess

        ax.clear()
        text = ax.text(0.5, 0.5, letter, fontsize=fontsize, fontname=fontname,
                        ha='center', va='center')

        plt.draw()  # Render the figure
        bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_height = bbox.height  # Bounding box height in pixels
        bbox_width = bbox.width


        fontsize_h = fontsize*(target_size / bbox_height)
        fontsize_w = fontsize*(target_size / bbox_width)
        
        #print(fontsize)
        fontsize = min(fontsize_h, fontsize_w)

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

    except RuntimeError:
        print('BAD')


def generator(font_list):
    # A = 65
    # a = 97

    letters_range = list(chain(range(65, 65+26), range(97, 97+26)))

    for f in font_list:
        print(f)
        for n in letters_range:
            plot_scaled_letter(unicode_num=n, fontname=f)

def char_images_reader(output = 'table', letter = 0):
    font_names = []
    char_numbers = []
    char_images = []

    if letter == 0:
        pattern = r"output\\([\w\s]+)_(\d+)\.png"
    else:
        pattern = r"output\\([\w\s]+)_(" + str(letter) + r")\.png"

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
    
    cases = np.where(np.asarray(char_numbers)<91, 1, 0)
    df.insert(2, 'case', cases)

    letters = list(map(chr, char_numbers))
    df.insert(3, 'letter', letters)

    df.to_csv(output + str(letter) + '.csv', index=False)

fonts = pd.read_csv('font_list.csv')['font_name']


# generate the font picture folder
#generator(fonts)


# reade the folder and generate a csv
#char_images_reader()

#plot_scaled_letter(unicode_num=97, fontname="Agency FB", target_size=20, dpi=30, show = True)
