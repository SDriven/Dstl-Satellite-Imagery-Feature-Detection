# Creating training and validation splits for the satellite data.

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random
import tifffile as tiff
import os
import logging
from shapely.wkt import loads
import cv2
import pandas as pd

DF = pd.read_csv('../data/train_wkt_v4.csv')
GS = pd.read_csv('../data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv('../data/sample_submission.csv')
CROP_SIZE = 160

def M(image_id, dims=20, size=1600):
    """
    Loads the tiff-files with different number of bands.
    """
    if dims==3:
        filename = "../data/three_band/{}.tif".format(
            image_id)
        img = tiff.imread(filename)
        img = np.rollaxis(img, 0, 3)
        img = cv2.resize(img, (size, size))
        return img
    elif dims==8:
        filename = "../data/sixteen_band/{}_M.tif".format(
            image_id)
        img = tiff.imread(filename)
        img = np.rollaxis(img, 0, 3)
        img = cv2.resize(img, (size, size))
    elif dims==20:
        img_M = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(
                image_id)), (1,2,0))
        img_M = cv2.resize(img_M, (size, size))

        img_A = np.transpose(tiff.imread("../data/sixteen_band/{}_A.tif".format(
            image_id)), (1,2,0))
        img_A = cv2.resize(img_A, (size, size))

        img_P = tiff.imread("../data/sixteen_band/{}_P.tif".format(
            image_id))
        img_P = cv2.resize(img_P, (size, size))

        filename = "../data/three_band/{}.tif".format(
            image_id)
        img_RGB = tiff.imread(filename)
        img_RGB = np.rollaxis(img_RGB, 0, 3)
        img_RGB = cv2.resize(img_RGB, (size, size))

        img = np.zeros((img_RGB.shape[0], img_RGB.shape[1], dims), "float32")
        img[..., 0:3] = img_RGB
        img[..., 3] = img_P
        img[..., 4:12] = img_M
        img[..., 12:21] = img_A
    return img

def init_logging(filename, message="START"):
    """
    Creates a logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info("-------------------------------------------------------------------------------------")
    logger.warning(message)
    logger.info("-------------------------------------------------------------------------------------")
    return logger

def _convert_coordinates_to_raster(coords, img_size, xymax):
    """
    Resize the polygon coordinates to the specific resolution of an image.
    """
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def _get_xmax_ymin(grid_sizes_panda, imageId):
    """
    To resize the training polygons, we need these parameters for each image.
    """
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)

def _get_polygon_list(wkt_list_pandas, imageId, class_type):
    """
    Load the training polygons with shapely.
    """
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = loads(multipoly_def.values[0])
    return polygonList

def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    """
    Create lists of exterior and interior coordinates of polygons resized to a specific image resolution.
    """
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list

def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    """
    Creates a class mask (0 and 1s) from lists of exterior and interior polygon coordinates.
    """
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask

def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    """
    Outputs a specific class mask from the training images.
    """
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask

def show_image(im, ms, number, name=""):
    """
    Outputs a plot with multiple subplots showing the original crop in RGB and the masks for all 10 classes.
    """
    image = np.zeros((im.shape[0], im.shape[1], 3))
    image[:, :, 0] = im[:, :, 0]  # red
    image[:, :, 1] = im[:, :, 1]  # green
    image[:, :, 2] = im[:, :, 2]  # blue
    classes = ["Buildings", "Misc. structures", "Road", "Track", "Trees", "Crops", "Waterway",
               "Standing Water", "Vehicle Large", "Vehicle Small"]
    f, (axarr) = plt.subplots(3, 4, sharey=True, figsize=(20,15))
    counter = 0
    for j in range(3):
        for k in range(4):
            if (j==0) & (k==0):
                tiff.imshow(image, figure=f, subplot=axarr[0,0])
                plt.grid("off")
                plt.title("Raw Image", size=22)
                continue
            elif (j==2) & (k==3):
                pass
            else:
                msk = ms[:,:,counter]
                tiff.imshow(255*np.stack([msk,msk,msk]), figure=f, subplot=axarr[j,k])
                plt.grid("off")
                # plt.title(name, size=22)
                plt.title("{} Mask".format(classes[counter]), size=22)
                counter += 1
    plt.grid("off")
    os.makedirs("../plots", exists_ok=True)
    plt.savefig("../plots/Crop_{}_{}.png".format(number, name), bbox_inches="tight", pad_inches=1)
    plt.clf()
    plt.cla()
    plt.close()

def get_crops(img, msk, how_many=10, aug=True, output=False):
    """
    Takes random 160x160 crops from larger images and optionally augments them by flipping them horizontally and
    vertically. Optionally outputs crops to check their validity.
    """
    is2 = int(1.0 * CROP_SIZE)
    # Anzahl Pixel des Riesenbilds minus 160 = Mögliche Links-unten Startpunkt für 160er Crop
    xm, ym = img.shape[0] - is2, img.shape[1] - is2
    x, y = [], []
    number = 0
    for _ in range(how_many):
        number += 1
        # Random Startpunkt unten links vom Crop
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)
        # 160x160 Crop
        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]
        if aug:
            if random.uniform(0, 1) > 0.5:
                im = im[::-1]
                ms = ms[::-1]
            if random.uniform(0, 1) > 0.5:
                im = im[:, ::-1]
                ms = ms[:, ::-1]
        if output:
            if random.uniform(0, 1) > 0.9:
                rnd = random.uniform(0, 1)
                show_image(im, ms, number, name="{}".format(rnd))
        x.append(im)
        y.append(ms)
    x, y = np.transpose(x, (0, 3, 1, 2)), np.transpose(y, (0, 3, 1, 2))
    print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    return x, y

def create_train_and_eval_splits(logger, output=False, name="", denominator=1, aug=True, size=1600, dims=20):
    """
    We take the 25 train images, split each of them into 4 quarter images and use a representative sample of these
    for evaluation. Different kinds of images get oversampled to simulate the test distribution of images.
    """
    print("let's stick all imgs together")
    class_list = ["Buildings", "Misc. Manmade structures", "Road", "Track", "Trees", "Crops", "Waterway",
                  "Standing Water", "Vehicle Large", "Vehicle Small"]

    id_list = ["6060_2_3", "6110_1_2", "6110_3_1", "6110_4_0","6120_2_0","6120_2_2","6140_1_2","6140_3_1","6150_2_3",
               "6100_1_3","6100_2_2","6100_2_3","6170_0_4","6170_2_4","6170_4_1","6010_1_2","6010_4_2","6010_4_4",
               "6040_1_0","6040_1_3","6040_2_2","6040_4_4","6090_2_0","6160_2_1","6070_2_3"]
    N_Cls = 10
    specific_oversampling_dict = {"6060_2_3": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6110_1_2": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6110_3_1": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6110_4_0": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6120_2_0": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6120_2_2": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6140_1_2": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6140_3_1": {0: 65, 1: 65, 2: 65, 3: 65},
                                  "6150_2_3": {0: 65, 1: 65, 2: 65, 3: 65}, # steppe cities
                                  "6100_1_3": {0: 25, 1: 25, 2: 25, 3: 25},
                                  "6100_2_2": {0: 25, 1: 25, 2: 25, 3: 25},
                                  "6100_2_3": {0: 25, 1: 25, 2: 25, 3: 25}, # green cities
                                  "6170_0_4": {0: 50, 1: 50, 2: 50, 3: 50},
                                  "6170_2_4": {0: 50, 1: 50, 2: 50, 3: 50},
                                  "6170_4_1": {0: 50, 1: 50, 2: 50, 3: 50}, # purple trees
                                  "6010_1_2": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6010_4_2": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6010_4_4": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6040_1_0": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6040_1_3": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6040_2_2": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6040_4_4": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6160_2_1": {0: 30, 1: 30, 2: 30, 3: 30},
                                  "6090_2_0": {0: 30, 1: 30, 2: 30, 3: 30}, # single trees, tracks
                                  "6070_2_3": {0: 200, 1: 100, 2: 150, 3: 150} } # forest river
    # because we are always taking 160x160 crops we can sample more crops if we are using images with higher resolution
    # denominator scales the amount of crops sampled.
    for k, v in specific_oversampling_dict.items():
        for k2, v2 in specific_oversampling_dict[k].items():
            specific_oversampling_dict[k][k2] = int(v2/denominator)
    # preselected quarter images for training. 0 = top left quarter, 1 = top right, 2 = bottom left, 3 = bottom right
    train_quarter_dict = {"6060_2_3": [0,2,3], "6110_1_2": [1,2,3], "6110_3_1":[1,2,3], "6110_4_0":[0,2,3],
                          "6120_2_0": [0,1,3],"6120_2_2":[1,2,3],"6140_1_2":[0,1,3],"6140_3_1":[0,2,3],
                          "6150_2_3":[0,1,3], "6100_1_3":[0,1,2,3],"6100_2_2":[1,2,3],"6100_2_3":[1,2,3],
                          "6170_0_4":[0,1,2],"6170_2_4":[0,2,3],"6170_4_1":[0,1,3],"6010_1_2":[0,1,2,3],
                          "6010_4_2":[0,1,2,3],"6010_4_4":[0,2,3], "6040_1_0":[0,2,3],"6040_1_3":[1,2,3],
                          "6040_2_2":[1,2,3],"6040_4_4":[0,1,3],"6090_2_0":[0,1,2],"6160_2_1":[0,1,2,3],
                          "6070_2_3":[0,1,2,3]}
    # preselected quarter images for evaluation. 0 = top left quarter, 1 = top right, 2 = bottom left, 3 = bottom right
    eval_quarter_dict = {"6060_2_3": [1], "6110_1_2": [0], "6110_3_1": [0], "6110_4_0": [1], "6120_2_0": [2],
                         "6120_2_2": [0], "6140_1_2": [2], "6140_3_1": [1], "6150_2_3": [2], "6100_1_3": [],
                         "6100_2_2": [0], "6100_2_3": [0], "6170_0_4": [3], "6170_2_4": [1], "6170_4_1": [2],
                         "6010_1_2": [], "6010_4_2": [], "6010_4_4": [1], "6040_1_0": [1], "6040_1_3": [0],
                          "6040_2_2": [0], "6040_4_4": [2], "6090_2_0": [3], "6160_2_1": [], "6070_2_3": [0, 1, 2, 3]}
    x, y = np.zeros((0,dims,160,160)).astype(np.float32), np.zeros((0,N_Cls,160,160)).astype(np.float32)
    for id in id_list:
        print(id)
        m = M(id, dims=dims, size=size)
        print(m.shape[0], m.shape[1])
        # upper left quarter of picture
        if 0 in train_quarter_dict[id]:
            img = m[:int(np.floor(m.shape[0]/2)), :int(np.floor(m.shape[1]/2)), :]
            mask = np.zeros((img.shape[0], img.shape[1], N_Cls))
            for z in range(10):
                mask[:,:,z] = generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                                         :int(np.floor(m.shape[0] / 2)), :int(np.floor(m.shape[1] / 2))]
            x_crops, y_crops = get_crops(img, mask, how_many=specific_oversampling_dict[id][0], output=output,
                                         aug=aug)
            x = np.concatenate((x, x_crops))
            y = np.concatenate((y, y_crops))
        # upper right quarter of picture
        if 1 in train_quarter_dict[id]:
            img = m[:int(np.floor(m.shape[0] / 2)), int(np.floor(m.shape[1] / 2)):]
            mask = np.zeros((img.shape[0], img.shape[1], N_Cls))
            for z in range(10):
                mask[:,:,z] = generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                              :int(np.floor(m.shape[0] / 2)), int(np.floor(m.shape[1] / 2)):]
            x_crops, y_crops = get_crops(img, mask, how_many=specific_oversampling_dict[id][1], output=output,
                                         aug=aug)
            x = np.concatenate((x, x_crops))
            y = np.concatenate((y, y_crops))
        # lower left quarter of picture
        if 2 in train_quarter_dict[id]:
            img = m[int(np.floor(m.shape[0] / 2)):, :int(np.floor(m.shape[1] / 2))]
            mask = np.zeros((img.shape[0], img.shape[1], N_Cls))
            for z in range(10):
                mask[:,:,z] = generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                              int(np.floor(m.shape[0] / 2)):, :int(np.floor(m.shape[1] / 2))]
            x_crops, y_crops = get_crops(img, mask, how_many=specific_oversampling_dict[id][2], output=output,
                                         aug=aug)
            x = np.concatenate((x, x_crops))
            y = np.concatenate((y, y_crops))
        # lower right quarter of picture
        if 3 in train_quarter_dict[id]:
            img = m[int(np.floor(m.shape[0] / 2)):, int(np.floor(m.shape[1] / 2)):]
            mask = np.zeros((img.shape[0], img.shape[1], N_Cls))
            for z in range(10):
                mask[:,:,z] = generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                              int(np.floor(m.shape[0] / 2)):, int(np.floor(m.shape[1] / 2)):]
            x_crops, y_crops = get_crops(img, mask, how_many=specific_oversampling_dict[id][3], output=output,
                                         aug=aug)
            x = np.concatenate((x, x_crops))
            y = np.concatenate((y, y_crops))
        print(x.shape)
        print(y.shape)
    os.makedirs("../data", exists_ok=True)
    np.save('../data/x_trn_{}_{}bands'.format(name, dims), x)
    np.save('../data/y_trn_{}_{}bands'.format(name, dims), y)

    print("{} BANDS".format(dims))
    for z in range(10):
        print("{:.4f}% Class {} in train set".format(
            100 * y[:,z].sum() / (y.shape[0] * y.shape[2] * y.shape[3]), class_list[z]))
        logger.info("{:.4f}% Class {} in train set".format(
            100 * y[:,z].sum() / (y.shape[0] * y.shape[2] * y.shape[3]), class_list[z]))
    print("-------------------------------------------------------------------------------------")
    logger.info("-------------------------------------------------------------------------------------")
    for z in range(10):
        print("{:.4f}% Class {} in small train set".format(
            100 * y[::10,z].sum() / (y[::10].shape[0] * y[::10].shape[2] * y[::10].shape[3]), class_list[z]))
        logger.info("{:.4f}% Class {} in small train set".format(
            100 * y[::10,z].sum() / (y[::10].shape[0] * y[::10].shape[2] * y[::10].shape[3]), class_list[z]))

    # FÜR EVAL
    x, y = np.zeros((0,dims,160,160)).astype(np.float32), np.zeros((0,N_Cls,160,160)).astype(np.float32)
    for id in id_list:
        m = M(id, dims=dims, size=size)
        # upper left quarter of picture
        if 0 in eval_quarter_dict[id]:
            img = m[:int(np.floor(m.shape[0]/2)), :int(np.floor(m.shape[1]/2))]
            mask = np.zeros((N_Cls, img.shape[0], img.shape[1]))
            for z in range(10):
                mask[z,:,:] =  generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                                         :int(np.floor(m.shape[0] / 2)), :int(np.floor(m.shape[1] / 2))]
            cnv = np.zeros((int(size/2), int(size/2), dims)).astype(np.float32)
            msk = np.zeros((N_Cls, int(size/2), int(size/2))).astype(np.float32)
            cnv[:img.shape[0], :img.shape[1], :] = img
            msk[:, :img.shape[0], :img.shape[1]] = mask
            line_x, line_y = [], []
            for i in range(int(size/(2*160))):
                for j in range(int(size/(2*160))):
                    line_x.append(cnv[i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
                    line_y.append(msk[:, i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
            x_ = np.transpose(line_x, (0, 3, 1, 2))
            y_ = np.transpose(line_y, (0, 1, 2, 3))
            x = np.concatenate((x, x_))
            y = np.concatenate((y, y_))
        # upper right quarter of picture
        if 1 in eval_quarter_dict[id]:
            img = m[:int(np.floor(m.shape[0] / 2)), int(np.floor(m.shape[1] / 2)):]
            mask = np.zeros((N_Cls, img.shape[0], img.shape[1]))
            for z in range(10):
                mask[z,:,:] =  generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                               :int(np.floor(m.shape[0] / 2)), int(np.floor(m.shape[1] / 2)):]
            cnv = np.zeros((int(size/2), int(size/2), dims)).astype(np.float32)
            msk = np.zeros((N_Cls, int(size/2), int(size/2))).astype(np.float32)
            cnv[:img.shape[0], :img.shape[1], :] = img
            msk[:, :img.shape[0], :img.shape[1]] = mask
            line_x, line_y = [], []
            for i in range(int(size/(2*160))):
                for j in range(int(size/(2*160))):
                    line_x.append(cnv[i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
                    line_y.append(msk[:, i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
            # x_ = 2 * np.transpose(line_x, (0, 3, 1, 2)) - 1
            x_ = np.transpose(line_x, (0, 3, 1, 2))
            y_ = np.transpose(line_y, (0, 1, 2, 3))
            x = np.concatenate((x, x_))
            y = np.concatenate((y, y_))
        # lower left quarter of picture
        if 2 in eval_quarter_dict[id]:
            img = m[int(np.floor(m.shape[0] / 2)):, :int(np.floor(m.shape[1] / 2))]
            mask = np.zeros((N_Cls, img.shape[0], img.shape[1]))
            for z in range(10):
                mask[z, :, :] = generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                                int(np.floor(m.shape[0] / 2)):, :int(np.floor(m.shape[1] / 2))]
            cnv = np.zeros((int(size/2), int(size/2), dims)).astype(np.float32)
            msk = np.zeros((N_Cls, int(size/2), int(size/2))).astype(np.float32)
            cnv[:img.shape[0], :img.shape[1], :] = img
            msk[:, :img.shape[0], :img.shape[1]] = mask
            line_x, line_y = [], []
            for i in range(int(size/(2*160))):
                for j in range(int(size/(2*160))):
                    line_x.append(cnv[i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
                    line_y.append(msk[:, i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
            x_ = np.transpose(line_x, (0, 3, 1, 2))
            y_ = np.transpose(line_y, (0, 1, 2, 3))
            x = np.concatenate((x, x_))
            y = np.concatenate((y, y_))
        # lower right quarter of picture
        if 3 in eval_quarter_dict[id]:
            img = m[int(np.floor(m.shape[0] / 2)):, int(np.floor(m.shape[1] / 2)):]
            mask = np.zeros((N_Cls, img.shape[0], img.shape[1]))
            for z in range(10):
                mask[z, :, :] = generate_mask_for_image_and_class((m.shape[0], m.shape[1]), id, z + 1)[
                                int(np.floor(m.shape[0] / 2)):, int(np.floor(m.shape[1] / 2)):]
            cnv = np.zeros((int(size/2), int(size/2), dims)).astype(np.float32)
            msk = np.zeros((N_Cls, int(size/2), int(size/2))).astype(np.float32)
            cnv[:img.shape[0], :img.shape[1], :] = img
            msk[:, :img.shape[0], :img.shape[1]] = mask
            line_x, line_y = [], []
            for i in range(int(size/(2*160))):
                for j in range(int(size/(2*160))):
                    line_x.append(cnv[i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
                    line_y.append(msk[:, i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])
            x_ = np.transpose(line_x, (0, 3, 1, 2))
            y_ = np.transpose(line_y, (0, 1, 2, 3))
            x = np.concatenate((x, x_))
            y = np.concatenate((y, y_))
    print("-------------------------------------------------------------------------------------")
    logger.info("-------------------------------------------------------------------------------------")
    for z in range(10):
        print("{:.4f}% Class {} in eval set".format(100*y[:,z].sum()/(y.shape[0]*160*160), class_list[z]))
        logger.info("{:.4f}% Class {} in eval set".format(100*y[:,z].sum()/(y.shape[0]*160*160), class_list[z]))
    np.save('../data/x_eval_{}_{}bands'.format(name, dims), x)
    np.save('../data/y_eval_{}_{}bands'.format(name, dims), y)

if __name__ == "__main__":
    os.makedirs("../logs", exist_ok=True)
    logger = init_logging("../logs/{}.log".format(datetime.now().strftme("%d-%m-%y")),
                          "START: Creating train/valid splits")
    create_train_and_eval_splits(logger, output=False, name="1600_denom.5", denominator=.5, aug=False, size=1600,
                                 dims=20)