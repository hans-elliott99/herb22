import tensorflow as tf



def prep_input_image(image, resize_factor=[380, 380]):
    # Normalize to ImageNet (the base model of the ResNet)
    image -= tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=image.dtype)
    # Crop towards central of image (crop out borders)
    image = tf.image.central_crop(image, central_fraction = 0.9)
    # Resize
    image = tf.image.resize(image, size = resize_factor) 
    return image


def normalize_to_ImageNet(image): # Preprocess with Imagenet's mean and stddev:
    image -= tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=image.dtype)
    return image






def augment_images(image, label, resize_factor):
    # Commented out lines that correspond to augmentation for generaliztion purposes
    max_angle=tf.constant(np.pi/6)
    #img = tf.image.random_flip_left_right(image) 
    #img = tfa.image.rotate(img, angles=max_angle*tf.random.uniform([1], minval=-1, maxval=1, dtype=tf.dtypes.float32)) # added random rotation, 30 degrees each side
    img = tf.image.central_crop(image, central_fraction = 0.9)
    img = tf.image.resize( img, size = resize_factor)
    return img, label


# this might be the appropriate fn
def load_image_and_resize(full_path, RESIZE, central_crop_frac = 0.9):
    
    raw = tf.io.read_file(full_path)
    img = tf.io.decode_image(raw)
    img = tf.image.central_crop(img, central_fraction = central_crop_frac)
    img = tf.image.resize(img, size = RESIZE)
    return(img)
