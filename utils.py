from PIL import Image


def load_image(image_path, transform=None, reshape=True):
    image = Image.open(image_path)
    if reshape:
        image = image.resize([224, 224], Image.LANCZOS)

    # image = Image.open(image_path)
    # plt.imshow(np.asarray(image))

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image
