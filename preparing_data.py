import os
import re
import random
import shutil
import scipy.misc
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from skimage.draw import circle
import matplotlib.pyplot as plt
import mritopng
from glob import glob
import cv2

# Function for reading PGM files
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

def convert_dcm_to_png(root_dir, source_dir):
    if not os.path.exists(os.path.join(root_dir, 'png')):
        os.makedirs(os.path.join(root_dir, 'png'))
    dcm_data_path = glob(os.path.join(root_dir, source_dir, '*.dcm'))
    for idx in range(np.shape(dcm_data_path)[0]):
        complit_data_name = re.split('/|[.]|\\\\', dcm_data_path[idx])[-2]
        data_name = re.split('/|[_]', complit_data_name)[0]
        print(os.path.join(root_dir, source_dir, complit_data_name, '.dcm'))
        mritopng.convert_file(dcm_data_path[idx],os.path.join(root_dir, 'png', data_name + '.png'))

def frac_eq_to(image, value=0):
    return (image == value).sum() / float(np.prod(image.shape))


def extract_patches(image, patchshape, overlap_allowed=0.1, cropvalue=None, crop_fraction_allowed=0.1):
    """
    Given an image, extract patches of a given shape with a certain
    amount of allowed overlap between patches, using a heuristic to
    ensure maximum coverage.
    If cropvalue is specified, it is treated as a flag denoting a pixel
    that has been cropped. Patch will be rejected if it has more than
    crop_fraction_allowed * prod(patchshape) pixels equal to cropvalue.
    Likewise, patches will be rejected for having more overlap_allowed
    fraction of their pixels contained in a patch already selected.
    """
    jump_cols = int(patchshape[1] * overlap_allowed)
    jump_rows = int(patchshape[0] * overlap_allowed)

    # Restrict ourselves to the rectangle containing non-cropped pixels
    if cropvalue is not None:
        rows, cols = np.where(image != cropvalue)
        rows.sort()
        cols.sort()
        active = image[rows[0]:rows[-1], cols[0]:cols[-1]]
    else:
        active = image

    rowstart = 0
    colstart = 0

    # Array tracking where we've already taken patches.
    covered = np.zeros(active.shape, dtype=bool)
    patches = []
    regions = []
    while rowstart <= active.shape[0] - patchshape[0]:
        # Record whether or not e've found a patch in this row,
        # so we know whether to skip ahead.
        got_a_patch_this_row = False
        colstart = 0
        while colstart <= active.shape[1] - patchshape[1]:
            # Slice tuple indexing the region of our proposed patch
            region = (slice(rowstart, rowstart + patchshape[0]),
                      slice(colstart, colstart + patchshape[1]))

            # The actual pixels in that region.
            patch = active[region]

            # The current mask value for that region.
            cover_p = covered[region]
            if cropvalue is None or \
                    frac_eq_to(patch, cropvalue) <= crop_fraction_allowed and \
                    frac_eq_to(cover_p, True) <= overlap_allowed:
                # Accept the patch.
                patches.append(patch)
                regions.append(region)
                # Mask the area.
                covered[region] = True

                # Jump ahead in the x direction.
                colstart += jump_cols
                got_a_patch_this_row = True
                # print "Got a patch at %d, %d" % (rowstart, colstart)
            else:
                # Otherwise, shift window across by one pixel.
                colstart += 1

        if got_a_patch_this_row:
            # Jump ahead in the y direction.
            rowstart += jump_rows
        else:
            # Otherwise, shift the window down by one pixel.
            rowstart += 1

    # Return a 3D array of the patches with the patch index as the first
    # dimension (so that patch pixels stay contiguous in memory, in a
    # C-ordered array).
    return np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0),regions


def filter_patches(patches, label, min_mean=50, min_std=1.0):
    """
    Filter patches by some criterion on their mean and variance.

    Takes patches, a 3-dimensional stack of image patches (where
    the first dimension indexes the patch), and a minimum
    mean, standard deviation and label.
    Returns:
        A stack of all the normal patches that satisfy both of these criteria
        A stack of all the normal patches that do not satisfy both of these criteria (Background pathes)
        A stack of all the abnormal patches.
    """
    patchdim = np.prod(patches.shape[1:])
    patchvectors = patches.reshape(patches.shape[0], patchdim)
    means = patchvectors.mean(axis=1)
    stdevs = patchvectors.std(axis=1)

    normal_patches_indices = (means >= min_mean) & (stdevs >= min_std) & (label == 0)
    background_patches_indices = ((means < min_mean) & (stdevs < min_std)) & (label == 0)
    abnormal_patches_indices = (label == 1)
    return patches[normal_patches_indices], patches[background_patches_indices], patches[abnormal_patches_indices]


def change_coordinates_axis(source_image, center_x, center_y, radius):
    return np.shape(source_image)[1] - center_y, center_x, radius


def create_mask(image, center_x, center_y, radius):
    width, height = np.shape(image)
    mask = np.zeros((width, height), dtype=np.uint8)
    rr, cc = circle(center_x, center_y, radius)
    mask[rr, cc] = 1

    return mask

def extract_patches_label(patches, mask_patches, ano_per=50.0):
    num_patches, _, _ = np.shape(patches)
    label = np.zeros((num_patches), dtype=np.uint8)
    per_ano_patches = np.zeros((num_patches), dtype=np.uint8)
    max_per_ano_patches = 0.0
    for idx in range(num_patches):
        unique, counts = np.unique(mask_patches[idx], return_counts=True)
        if unique[0] == 0 and np.ndarray.max(unique) == 1:
            per_ano_patches[idx] = (counts[1] / (counts[0] + counts[1])) * 100.0
        elif unique[0] == 0 and np.ndarray.max(unique) == 0:
            per_ano_patches[idx] = 0.0
        elif unique[0] == 1:
            per_ano_patches[idx] = 100.0

        if(per_ano_patches[idx]>=max_per_ano_patches):
            max_per_ano_patches = per_ano_patches[idx]
    print(max_per_ano_patches)
    for idx in range(num_patches):

        if np.ndarray.max(mask_patches[idx]) != 0 and per_ano_patches[idx] >= max_per_ano_patches and per_ano_patches[idx] >=ano_per:
            label[idx] = 1
        elif np.ndarray.max(mask_patches[idx]) != 0 and (per_ano_patches[idx] < max_per_ano_patches or per_ano_patches[idx] <ano_per):
            label[idx] = -1

        idx += 1
    return label

def crop_center(image, c_x, c_y, radius, margin, outpu_size):
    width, height = np.shape(image)
    x_min = c_x - radius - margin
    x_max = c_x + radius + margin
    y_min = c_y - radius - margin
    y_max = c_y + radius + margin
    if (x_min < 0):
        x_min = 0
    if (x_max > width - 1):
        x_max = width - 1
    if (y_min < 0):
        y_min = 0
    if (y_max > height - 1):
        y_max = height - 1
    return scipy.misc.imresize(image[x_min:x_max, y_min:y_max], outpu_size, interp='bilinear', mode=None)

def reshape_images_patches(images_patches):
    num_full_image = np.shape(images_patches)[0]
    output_patches = []
    for i in range(num_full_image):
        num_patches = np.shape(images_patches[i])[0]
        for j in range(num_patches):
            output_patches.append(images_patches[i][j])
    return output_patches

def adaptive_histogram_equalization(img):
    # create a CLAHE object (Arguments are optional).
    img = np.uint8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1

def save_mias_patches(normal_patches, abnormal_patches, background_patches, root_dir, train_dir, test_dir):
    num_abnormal_patches = np.shape(abnormal_patches)[0]
    num_normal_patches = np.shape(normal_patches)[0]
    num_background = max(50, int(num_normal_patches / 40))
    random.shuffle(normal_patches)
    random.shuffle(background_patches)
    counter = 0
    for idx in range(num_abnormal_patches):
        if np.shape(abnormal_patches[idx]) != np.shape(normal_patches[idx]):
            abnormal_patches[idx] = scipy.misc.imresize(abnormal_patches[idx], np.shape(normal_patches[idx]),
                                                        interp='nearest', mode='L')
        plt.imsave(os.path.join(root_dir, test_dir, 'abnormal') + '/' + 'abnormal-'
                   + str(idx) + '.png', abnormal_patches[idx], cmap=cm.gray)
        plt.imsave(os.path.join(root_dir, test_dir, 'normal') + '/' + 'normal-'
                   + str(idx) + '.png', normal_patches[idx], cmap=cm.gray)

    for idx in range(num_abnormal_patches, num_normal_patches):
        plt.imsave(os.path.join(root_dir, train_dir) + '/' + 'normal-'
                   + str(counter) + '.png', normal_patches[idx], cmap=cm.gray)
        counter += 1
    for idx in range(num_background):
        plt.imsave(os.path.join(root_dir, train_dir) + '/' + 'normal-'
                   + str(counter) + '.png', background_patches[idx], cmap=cm.gray)
        counter += 1

def save_abnorm_inbreast_patches(mass_patches, root_dir, test_dir, patch_size):
    num_mass_patches = np.shape(mass_patches)[0]

    for idx in range(num_mass_patches):

        if np.shape(mass_patches[idx]) != patch_size:
            mass_patches[idx] = scipy.misc.imresize(mass_patches[idx],patch_size,
                                                        interp='nearest', mode='L')
        plt.imsave(os.path.join(root_dir, test_dir, 'abnormal') + '/' + 'mass-'
                   + str(idx) + '.png', mass_patches[idx], cmap=cm.gray)

counter_normal_inbreast=0

def save_normal_inbreast_patches(normal_patches,background_patches,root_dir):
    global counter_normal_inbreast

    num_normal_patches = np.shape(normal_patches)[0]
    print(np.shape(normal_patches))
    num_background = max(1, int(num_normal_patches / 40))
    print(np.shape(background_patches))
    random.shuffle(normal_patches)
    random.shuffle(background_patches)

    for idx in range(num_normal_patches):
        plt.imsave(os.path.join(root_dir, 'normal_tmp') + '/' + 'normal-'
                   + str(counter_normal_inbreast) + '.png', normal_patches[idx], cmap=cm.gray)
        counter_normal_inbreast += 1
    for idx in range(num_background):
        plt.imsave(os.path.join(root_dir, 'normal_tmp') + '/' + 'normal-'
                   + str(counter_normal_inbreast) + '.png', background_patches[idx], cmap=cm.gray)
        counter_normal_inbreast += 1


def split_to_test_train_inbreast(root_dir, train_dir, test_dir):
    normal_data_path = glob(os.path.join(root_dir, 'normal_tmp', '*.png'))
    abnormal_data_path = glob(os.path.join(root_dir, test_dir, 'abnormal', '*.png'))
    num_abnormal = np.shape(abnormal_data_path)[0]
    num_normal = np.shape(normal_data_path)[0]

    rand_num_list = random.sample(range(0, num_normal), num_abnormal)
    counter_test = 0
    counter_train = 0
    for idx in range(num_normal):
        src_image = scipy.misc.imread(normal_data_path[idx])
        if idx in rand_num_list:
            plt.imsave(os.path.join(root_dir, test_dir, 'normal') + '/' + 'normal-'
                       + str(counter_test) + '.png', src_image, cmap=cm.gray)
            counter_test+=1
        else:
            plt.imsave(os.path.join(root_dir, train_dir) + '/' + 'normal-'
                       + str(counter_train) + '.png', src_image, cmap=cm.gray)
            counter_train+=1
    #shutil.rmtree(os.path.join(root_dir, 'normal_tmp'))


def preparing_mias_data(root_dir, source_dir, train_dir, test_dir, patchsize, overlap_allowed=0.4, cropvalue=None,
                        crop_fraction_allowed=1, min_mean=50, min_std=1, margin=20):
    print("Preparing Mias dataset ...")
    if os.path.exists(os.path.join(root_dir, train_dir)):
        shutil.rmtree(os.path.join(root_dir, train_dir))
    os.makedirs(os.path.join(root_dir, train_dir))
    if os.path.exists(os.path.join(root_dir, test_dir)):
        shutil.rmtree(os.path.join(root_dir, test_dir))
    os.makedirs(os.path.join(root_dir, test_dir))
    os.makedirs(os.path.join(root_dir, test_dir, 'normal'))
    os.makedirs(os.path.join(root_dir, test_dir, 'abnormal'))
    # read csvFile
    all_normal_patches = []
    all_abnormal_patches = []
    all_background_patches = []
    data = pd.read_csv(os.path.join(root_dir, source_dir, 'mias.csv'))
    patch_size = [patchsize, patchsize]
    for i, row in data.iterrows():

        # Read source image
        source_image = read_pgm(os.path.join(root_dir, source_dir, row['reference_number'] + '.pgm'))
        source_image = adaptive_histogram_equalization(source_image)
        image_patches, _ = extract_patches(source_image, patch_size, overlap_allowed=overlap_allowed, cropvalue=cropvalue,
                                        crop_fraction_allowed=crop_fraction_allowed)

        if row['abnormality_class'] == 'NORM':
            label = np.zeros((np.shape(image_patches)[0]), dtype=np.uint8)
            image_patches, background_patches,_ = filter_patches(image_patches, label, min_mean=min_mean, min_std=min_std)

            all_normal_patches.append(image_patches)

            all_background_patches.append(background_patches)
        else:
            if (np.isnan(row['radius'])):
                continue
            c_x, c_y, radius = change_coordinates_axis(source_image, int(row['x']), int(row['y']), int(row['radius']))
            mask = create_mask(source_image, c_x, c_y, radius)
            mask_patches, _ = extract_patches(mask, patch_size, overlap_allowed=overlap_allowed, cropvalue=cropvalue,
                                           crop_fraction_allowed=crop_fraction_allowed)
            label = extract_patches_label(image_patches, mask_patches)
            image_patches, background_patches,_ = filter_patches(image_patches, label, min_mean=min_mean, min_std=min_std)
            all_normal_patches.append(image_patches)
            all_background_patches.append(background_patches)
            abnormal_image = crop_center(source_image, c_x, c_y, radius, margin=margin, outpu_size=patch_size)
            all_abnormal_patches.append(abnormal_image)
    all_normal_patches = reshape_images_patches(all_normal_patches)
    all_background_patches = reshape_images_patches(all_background_patches)
    save_mias_patches(all_normal_patches, all_abnormal_patches, all_background_patches, root_dir, train_dir, test_dir)
    print("end ...")

def preparing_inbreast_data(root_dir , source_dir , train_dir , test_dir , patchsize , overlap_allowed=0.4 ,cropvalue=None ,
                            crop_fraction_allowed=1 , min_mean=20 , min_std=1 , margin=20):
    print("Preparing INBreast dataset ...")

    if os.path.exists(os.path.join(root_dir , train_dir)):
        shutil.rmtree(os.path.join(root_dir , train_dir))
    os.makedirs(os.path.join(root_dir , train_dir))
    if os.path.exists(os.path.join(root_dir , test_dir)):
        shutil.rmtree(os.path.join(root_dir , test_dir))
    os.makedirs(os.path.join(root_dir , test_dir))
    os.makedirs(os.path.join(root_dir , test_dir , 'normal'))
    os.makedirs(os.path.join(root_dir , test_dir , 'abnormal'))

    if os.path.exists(os.path.join(root_dir , 'normal_tmp')):
        shutil.rmtree(os.path.join(root_dir , 'normal_tmp'))
    os.makedirs(os.path.join(root_dir , 'normal_tmp'))
    convert_dcm_to_png(root_dir , source_dir)

    data_description = pd.read_excel(os.path.join(root_dir , source_dir , 'INbreast.xls'))
    patch_size = [patchsize , patchsize]

    all_mass_patches = []

    for i , row in data_description.iterrows():
        if str(row['File Name']) != 'nan':

            print("--image " + str(i + 1) + " --image name: " + str(row['File Name']))
            # Read source image
            source_image = scipy.misc.imread(os.path.join(root_dir , 'png' , str(int(row['File Name'])) + '.png'))
            source_image = adaptive_histogram_equalization(source_image)

            image_patches , _ = extract_patches(source_image , patch_size , overlap_allowed=overlap_allowed ,
                                                cropvalue=cropvalue ,
                                                crop_fraction_allowed=crop_fraction_allowed)
            print(row['Mass'] , row['Micros'])

            if row['Mass'] != 'X':
                label = np.zeros((np.shape(image_patches)[0]) , dtype=np.uint8)
                image_patches , background_patches , _ = filter_patches(image_patches , label , min_mean=min_mean ,
                                                                        min_std=min_std)
                save_normal_inbreast_patches(image_patches , background_patches , root_dir)
            else:
                if os.path.exists(os.path.join(root_dir , source_dir , 'mask' , 'MassSegmentationMasks' ,
                                               str(int(row['File Name'])) + '_mask.png')):
                    mask = cv2.imread(os.path.join(root_dir , source_dir , 'mask' , 'MassSegmentationMasks' ,
                                                   str(int(row['File Name'])) + '_mask.png') , 0) / 255

                    mask_patches , _ = extract_patches(mask , patch_size , overlap_allowed=overlap_allowed ,
                                                       cropvalue=cropvalue ,
                                                       crop_fraction_allowed=crop_fraction_allowed)

                    label = extract_patches_label(image_patches , mask_patches)

                    image_patches , background_patches , abnormal_patches = filter_patches(image_patches , label ,
                                                                                           min_mean=min_mean ,
                                                                                           min_std=min_std)
                    save_normal_inbreast_patches(image_patches , background_patches , root_dir)
                    all_mass_patches.append(abnormal_patches)
                else:
                    print("Error in " + 'MassSegmentationMasks ' , str(int(row['File Name'])) + '_mask.png')

    all_mass_patches = reshape_images_patches(all_mass_patches)

    save_abnorm_inbreast_patches(all_mass_patches , root_dir , test_dir , patch_size)

    split_to_test_train_inbreast(root_dir, train_dir, test_dir)

    print("end ...")
