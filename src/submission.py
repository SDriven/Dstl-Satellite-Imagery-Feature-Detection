# Predicting for 429 test images and creating a submission

from Preprocessing import M, init_logging
from training import get_unet
from datetime import datetime
import numpy as np
import cv2
import os
from shapely.geometry import MultiPolygon, Polygon
import shapely.affinity
from shapely.wkt import dumps
from collections import defaultdict
import pandas as pd


CROP_SIZE = 160
GS = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv('data/sample_submission.csv')
class_list = ["Buildings", "Misc. Manmade structures", "Road", "Track", "Trees", "Crops", "Waterway",
              "Standing Water", "Vehicle Large", "Vehicle Small"]

def predict_id(id, model, trs, dims, size=1600, mins=None, maxs=None, use_sample_weights=False, raw=False, means=None):
    """
    Predicts a single test image by predicting for all 160x160 crops in the larger image.
    """
    x = M(id, dims=dims, size=size)
    h, w = x.shape[0], x.shape[1]

    def min_max_normalize(bands, mins, maxs):
        out = np.zeros_like(bands).astype(np.float32)
        n = bands.shape[2]
        for i in range(n):
            a = 0  # np.min(band)
            b = 1  # np.max(band)
            c = mins[i]
            d = maxs[i]
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t
        return out.astype(np.float32)

    # Normalization: Scale with Min/Max
    x = min_max_normalize(x, mins, maxs)

    pixels = size
    rows = int(size / 160)
    cnv = np.zeros((pixels, pixels, dims)).astype(np.float32)
    prd = np.zeros((10, pixels, pixels)).astype(np.float32)
    cnv[:h, :w, :] = x

    line = []
    for i in range(0, rows):
        # we slide through 160x160 crops and append them for prediction after
        for j in range(0, rows):
            line.append(cnv[i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE])

    x = np.transpose(line, (0, 3, 1, 2))
    if means is not None:
        for k in range(dims):
            x[:,k] -= means[k]
    tmp = model.predict(x, batch_size=16)
    if use_sample_weights:
        # Output is (None, 160*160, 10), reshape to (None, 10, 160, 160)
        tmp = np.rollaxis(tmp, 2, 1)
        tmp = tmp.reshape(tmp.shape[0], 10, 160, 160)
    k = 0
    for i in range(rows):
        for j in range(rows):
            prd[:, i * CROP_SIZE:(i + 1) * CROP_SIZE, j * CROP_SIZE:(j + 1) * CROP_SIZE] = tmp[k]
            k += 1
    if raw:
        pass
    else:
        for i in range(10):
            prd[i] = prd[i] >= trs[i]
    return prd

def predict_test(max_score="", model=None, trs=None, size=1600, mins=None, maxs=None, dims=20, raw=True, means=None):
    """
    Creates and saves predictions for all test images.
    """
    print("Predicting test images with a model scoring {}".format(max_score))
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs, dims, size=size, mins=mins, maxs=maxs, raw=raw, means=means)
        np.save('msk/{}_{}'.format(max_score, id), msk)
        if i%20==0: print(i, id)

def mask_to_polygons(mask, epsilon=1, min_area=1.):
    """
    Create a Multipolygon from a mask of 0-1 pixels.
    """
    # find contours of mask of pixels
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

def make_submit(name, max_score, trs=None):
    """
    Creates the final submission by loading all raw predictions, creating 0-1 masks by thresholding them and
    creating Multipolygons from these masks.
    """
    print("make submission file")
    # Get the best scores to load the best predictions
    for idx, row in SB.iterrows():
        id = row[0]
        kls = row[1] - 1
        # Get the prediction from the respective class model and the best performing iteration of it
        msk = np.load('../msk/{}_{}.npy'.format(max_score, id))[kls]
        msk = msk >= trs[kls]
        # Create correctly sizes polygons for the submission file
        pred_polygons = mask_to_polygons(msk, epsilon=1, min_area=1)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]
        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)
        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))
        SB.iloc[idx, 2] = dumps(scaled_pred_polygons, rounding_precision=8)
        if SB.iloc[idx, 2]=="GEOMETRYCOLLECTION EMPTY":
            SB.iloc[idx, 2] = "MULTIPOLYGON EMPTY"
        if idx % 100 == 0: print(idx)
    os.makedirs("../subm", exist_ok=True)
    SB.to_csv('../subm/{}.csv.gz'.format(name), compression="gzip", index=False)

if __name__ == "__main__":
    logger = init_logging("../logs/{}.log".format(datetime.now().strftme("%d-%m-%y")),
                          "START: Submitting")
    # precomputed minimum and maximum values for all spectral bands
    mins = [55.0, 167.0, 99.0, 174.0, 182.0, 144.0, 158.0, 132.0, 61.0, 138.0, 160.0, 113.0, 672.0, 490.0, 435.0,
            391.0, 55.0, 168.0, 187.0, 55.0]
    maxs = [2047.0, 2047.0, 2047.0, 2040.0, 2035.0, 2047.0, 2047.0, 2047.0, 2047.0, 2047.0, 2047.0, 2047.0, 15410.0,
            16050.0, 16255.0, 16008.0, 15933.0, 15805.0, 15878.0, 15746.0]

    scores = [float(j[-6:]) for j in os.listdir("../weights")]
    max_score = "{:.4f}".format(max(scores))
    visual_name = "conv32_nobn_nodo_bs16_decay.97"
    model = get_unet(dims=20, conv_channel=32, bn=False, dropout=False, big=True, N_Cls=10)
    model.load_weights('../weights/unet_{}_{}'.format(visual_name, max_score))
    trs = np.load("../data/thresholds_unet_{}_{}.npy".format(visual_name, max_score))
    means = np.load("../data/means_{}.npy".format(visual_name))
    predict_test(max_score, model=model)
    make_submit("subm_1", max_score)