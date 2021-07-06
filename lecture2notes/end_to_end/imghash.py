import logging
import os

import imagehash
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_hash_func(hashmethod="phash"):
    """
    Returns a hash function from the ``imagehash`` library.

    Hash Methods:
        * ahash: Average hash
        * phash: Perceptual hash
        * dhash: Difference hash
        * whash-haar: Haar wavelet hash
        * whash-db4: Daubechies wavelet hash
    """
    if hashmethod == "ahash":
        hashfunc = imagehash.average_hash
    elif hashmethod == "phash":
        hashfunc = imagehash.phash
    elif hashmethod == "dhash":
        hashfunc = imagehash.dhash
    elif hashmethod == "whash-haar":
        hashfunc = imagehash.whash
    elif hashmethod == "whash-db4":
        hashfunc = lambda img: imagehash.whash(img, mode="db4")  # noqa: E731

    return hashfunc


def sort_by_duplicates(img_dir, hash_func="phash"):
    """Find duplicate images in a directory.

    Args:
        img_dir (str): path to folder containing images to scan for duplicates
        hash_func (str, optional): the hash function to use as given by
            :meth:`~lecture2notes.end_to_end.imghash.get_hash_func`. Defaults to "phash".

    Returns:
        [dict]: dictionary in format {image hash: image filenames}
    """
    logger.info("Identifying frames/slides that are potential duplicates")
    hashfunc = get_hash_func(hash_func)

    images = {}
    image_filenames = sorted(os.listdir(img_dir))

    for img in tqdm(
        sorted(image_filenames),
        desc="Img Hasher> Computing Hashes",
        total=len(image_filenames),
    ):
        current_img_path = os.path.join(img_dir, img)

        img_hash = hashfunc(Image.open(current_img_path))
        if img_hash in images:
            logger.debug("%s already exists as %s", img, " ".join(images[img_hash]))
        # store the image at with its hash as a key (add the image to the list for the respective key if
        # that key already exists)
        images[img_hash] = images.get(img_hash, []) + [img]

    return images


def remove_duplicates(img_dir, images):
    """Remove duplicate frames/slides from disk.

    Args:
        img_dir (str): path to directory containing image files
        images (dict): dictionary in format {image hash: image filenames}
            provided by :meth:`~lecture2notes.end_to_end.imghash.sort_by_duplicates`.
    """
    logger.info("Removing duplicate frames/slides from disk")
    for img_hash, img_paths in images.items():
        # if there is more than one image with the same path
        if len(img_paths) > 1:
            # remove all but the last image
            img_paths = sorted(img_paths)
            for img in img_paths[:-1]:
                logger.debug("Removing " + str(img))
                os.remove(os.path.join(img_dir, img))


# images = sort_by_duplicates("slide_clusters/best_samples")
# remove_duplicates("slide_clusters/best_samples", images)
