import os
import numpy as np
import imageio

# img_dir: file folder that contains imgs about to convert
def compose_gif(img_dir):

    img_paths = os.listdir(img_dir)
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(img_dir + path))
    # save to file folder ./gif
    imageio.mimsave("./gif/" + img_paths[-1] + ".gif", gif_images, fps=10)

compose_gif("./gifs/")
