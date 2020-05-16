import os
import logging
import imagehash
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_hash_func(hashmethod="phash"):
    """
    Hash Methods: 
        ahash:      Average hash
        phash:      Perceptual hash
        dhash:      Difference hash
        whash-haar: Haar wavelet hash
        whash-db4:  Daubechies wavelet hash
    """
    if hashmethod == 'ahash':
        hashfunc = imagehash.average_hash
    elif hashmethod == 'phash':
        hashfunc = imagehash.phash
    elif hashmethod == 'dhash':
        hashfunc = imagehash.dhash
    elif hashmethod == 'whash-haar':
        hashfunc = imagehash.whash
    elif hashmethod == 'whash-db4':
        hashfunc = lambda img: imagehash.whash(img, mode='db4')
    
    return hashfunc

def sort_by_duplicates(img_dir, hash_func="phash"):
    logger.info("Identifying frames/slides that are potential duplicates")
    hashfunc = get_hash_func(hash_func)

    images = {}
    image_filenames = sorted(os.listdir(img_dir))
    
    for img in tqdm(sorted(image_filenames), desc="Img Hasher> Computing Hashes", total=len(image_filenames)):
        current_img_path = os.path.join(img_dir, img)

        img_hash = hashfunc(Image.open(current_img_path))
        if img_hash in images:
            logger.debug(img, ' already exists as', ' '.join(images[img_hash]))
        # store the image at with its hash as a key (add the image to the list for the respective key if
        # that key already exists)
        images[img_hash] = images.get(img_hash, []) + [img]

    return images

def remove_duplicates(img_dir, images):
    logger.info("Removing duplicate frames/slides from disk")
    for img_hash, img_paths in images.items():
        # if there is more than one image with the same path
        if len(img_paths) > 1:
            # remove all but the last image
            img_paths = sorted(img_paths)
            for img in img_paths[:-1]:
                os.remove(os.path.join(img_dir, img))

# images = sort_by_duplicates("slide_clusters/best_samples")
# remove_duplicates("slide_clusters/best_samples", images)