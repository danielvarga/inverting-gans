import numpy as np
from PIL import Image


def merge_sets(arrays):
    size = arrays[0].shape[0]
    result = []
    for i in range(size):
        for array in arrays:
            assert array.shape[0] == size, "Incorrect length {} in the {}th array".format(array.shape[0], i)
            result.append(array[i])
    return np.array(result)


# assumes (0, 1) range
def plot_images(data, n_x, n_y, name, text=None):
    data = data[:n_x * n_y].copy()
    data = np.clip(data, 0, 1)

    (height, width, channel) = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    if channel == 1:
        mode = "L"
        data = data[:,:,:,0]
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    else:
        mode = "RGB"
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1, channel), dtype='uint8')
    for idx in range(n):
        x = idx % n_x
        y = idx // n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)
    fileName = name + ".png"
    print("Creating file " + fileName)
    if text is not None:
        img.text(10, 10, text)
    img.save(fileName)


def display_reconstructed(hourglass, images, name):
    images = images[:50].copy()
    recons = hourglass.predict(images)
    mergedSet = merge_sets([images, recons])
    plot_images(mergedSet, 10, 10, name)


# todo move to vis.py
def flattorus_visualization(decoder, name):
    grid_size = 30
    batch_size = 25
    x = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    y = x
    xx, yy = np.meshgrid(x, y)
    # cxc as in circle x circle, a Descartes product
    cxc = np.vstack([ xx.reshape(-1), yy.reshape(-1) ]).T
    assert cxc.shape == (grid_size*grid_size, 2)
    # turning two points on the circle into a point of the torus embedded in 4d:
    embedded = np.vstack([np.cos(cxc[:, 0]), np.sin(cxc[:, 0]), np.cos(cxc[:, 1]), np.sin(cxc[:, 1])]).T
    assert embedded.shape == (grid_size*grid_size, 4)
    plane = embedded.reshape((grid_size, grid_size, 4))
    images = []
    height, width, l_d = plane.shape
    x_decoded = decoder.predict(embedded, batch_size=batch_size)
    plot_images(x_decoded, grid_size, grid_size, name)
