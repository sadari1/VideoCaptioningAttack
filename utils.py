from PIL import Image

import pickle
from datetime import datetime


def load_image(image_path, transform=None, reshape=True):
    image = Image.open(image_path)
    if reshape:
        image = image.resize([224, 224], Image.LANCZOS)

    # image = Image.open(image_path)
    # plt.imshow(np.asarray(image))

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def get_time_stamp():
    date_object = datetime.now()
    return date_object.strftime('%m%d%y-%H%M%S')


def pickle_write(fpath, obj):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)

    return obj
