import cv2
import numpy as np
import os


def read_img(image_path):
    image = cv2.imread(cv2.samples.findFile(image_path))
    if image is None:
        print('Could not open or find the image: {}'.format(image_path))
        exit(0)
    return image


def change_contrast(image_path):
    factor_list = [-0.5, 0.0, 0.5, 2.0]
    image = read_img(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_lumi = hsv[..., 2].mean()

    img_BGRA = np.zeros(image.shape, dtype=image.dtype)
    img_BGRA[:, :, 0] = avg_lumi
    img_BGRA[:, :, 1] = avg_lumi
    img_BGRA[:, :, 2] = avg_lumi

    for alpha in factor_list:
        new_image = np.zeros(image.shape, image.dtype)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y, x, c] = np.clip((1 - alpha) * img_BGRA[y, x, c] + alpha * image[y, x, c], 0, 255)
        cv2.imwrite('output/c_contrast_{}.jpg'.format(str(alpha)), new_image)


def brighten(image_path):
    factor_list = [0.0, 0.5, 2.0]
    image = read_img(image_path)
    for br_fac in factor_list:
        bright_image = np.zeros(image.shape, image.dtype)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    bright_image[y, x, c] = np.clip(br_fac * image[y, x, c], 0, 255)
        cv2.imwrite('output/princeton_small_brightness_{}.jpg'.format(str(br_fac)), bright_image)


def change_saturation(image_path):
    factor_list = [-1.0, 0.0, 2.5]
    image = read_img(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for factor in factor_list:
        new_image = np.zeros(image.shape, image.dtype)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip((1-factor) * gray_img[y, x] + factor * image[y, x, c], 0, 255)

        cv2.imwrite('output/saturated_{}.jpg'.format(str(factor)), new_image)


def image_sharpen(image_path):
    image = read_img(image_path)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    # Putting a negative value for depth to indicate to use the
    # the same depth of the source image src.depth().
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    cv2.imwrite('output/sharpen.jpg', sharpened)


def blur(image_path):
    factor_list = [0.125, 0.5, 2, 8]
    image = read_img(image_path)
    for sigma in factor_list:
        blur_img = cv2.GaussianBlur(image, (7, 7), sigma)
        cv2.imwrite('output/blur_{}.jpg'.format(str(sigma)), blur_img)


def edge_detect(image_path):
    image = read_img(image_path)
    # Canny Edge detection
    edges = cv2.Canny(image, 100, 200)
    cv2.imwrite('output/edgedetect.jpg', edges)


def composite(background_img_path, foreground_img_path, alpha_img_path):
    foreground = read_img(foreground_img_path)
    background = read_img(background_img_path)
    alpha = read_img(alpha_img_path)
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    out_image = cv2.add(foreground, background)

    # Display image
    cv2.imwrite("output/composite.jpg", out_image)


if __name__ == '__main__':
    print("Processing Image Files. This may take 5-6 minutes. Please Wait...")
    change_contrast('input' + os.sep + 'c.jpg')
    brighten('input' + os.sep + 'princeton_small.jpg')
    blur('input' + os.sep + 'princeton_small.jpg')
    image_sharpen('input' + os.sep + 'princeton_small.jpg')
    edge_detect('input' + os.sep + 'princeton_small.jpg')
    change_saturation('input' + os.sep + 'scaleinput.jpg')
    composite(background_img_path='input' + os.sep + 'comp_background.jpg',
              foreground_img_path='input' + os.sep + 'comp_foreground.jpg',
              alpha_img_path='input' + os.sep + 'comp_mask.jpg')