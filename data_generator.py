import numpy as np 
from scipy.io import savemat
import cv2

delta_p = 10
delta_t = 10

image_size = 128
stripe_size = 18
square = 20

nbr_image = 200

synthetic_images = np.zeros((image_size, image_size, nbr_image))
synthetic_images[0:image_size - stripe_size, :, 0] = 255
# synthetic_images[0:square, :square, 0] = 255


for i in range(1, nbr_image):
    # synthetic_images[:, :, i] = np.roll(synthetic_images[:, :, i-1], 1, axis=0)
    synthetic_images[:image_size - stripe_size - i, :, i] = 255
    # synthetic_images[:, :, i] = np.roll(synthetic_images[:, :, i-1], 1, axis=0)
    # synthetic_images[:, :, i] = np.roll(synthetic_images[:, :, i], 1, axis=1)
    cv2.imshow("generated", synthetic_images[:, :, i])
    cv2.waitKey(10)

# Add noise
# synthetic_images = synthetic_images + np.random.normal(0, 10, synthetic_images.shape)

synthetic_events = np.zeros((0, 4))

# for i in range(1, nbr_image):

#     diff_image = synthetic_images[:, :, i] - synthetic_images[:, :, i-1]
#     changes = np.where(np.abs(diff_image) > delta_p)
#     new_synthetic_events = np.concatenate((changes[0].reshape(-1, 1), changes[1].reshape(-1, 1), np.sign(diff_image[changes]).reshape(-1, 1), i*delta_t*np.ones((changes[0].shape[0], 1))), axis = 1)
#     synthetic_events = np.concatenate((synthetic_events, new_synthetic_events), axis = 0)

# print(synthetic_events)
# savemat('synthetic_stripes.mat', {'data': synthetic_events})