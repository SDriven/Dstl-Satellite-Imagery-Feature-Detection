# Training the U-Net

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import time
from Preprocessing import M, init_logging
from datetime import datetime
import numpy as np
import os
# os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Activation, Permute
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K
K.set_image_dim_ordering("th")

CROP_SIZE = 160
class_list = ["Buildings", "Misc. Manmade structures", "Road", "Track", "Trees", "Crops", "Waterway",
              "Standing Water", "Vehicle Large", "Vehicle Small"]

def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    """
    Cuts of extreme values of spectral bands to visualize them better for the human eye.
    """
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:,:,i], lower_percent)
        d = np.percentile(bands[:,:,i], higher_percent)
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t
    return out.astype(np.uint8)

def get_unet(lr=1e-4, deep=True, dims=20, conv_channel=32, N_Cls=10, bn=False, use_sample_weights=True,
             init="glorot_uniform"):
    """
    Creates the U-Net in Keras
    """
    inputs = Input((dims, CROP_SIZE, CROP_SIZE))
    conv1 = Convolution2D(conv_channel, 3, 3, border_mode='same', init=init)(inputs)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Convolution2D(conv_channel, 3, 3, border_mode='same', init=init)(conv1)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(conv_channel * 2, 3, 3, border_mode='same', init=init)(pool1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Convolution2D(conv_channel * 2, 3, 3, border_mode='same', init=init)(conv2)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(conv_channel * 4, 3, 3, border_mode='same', init=init)(pool2)
    if bn:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Convolution2D(conv_channel * 4, 3, 3, border_mode='same', init=init)(conv3)
    if bn:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(conv_channel * 8, 3, 3, border_mode='same', init=init)(pool3)
    if bn:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = Convolution2D(conv_channel * 8, 3, 3, border_mode='same', init=init)(conv4)
    if bn:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    if deep:
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(conv_channel * 16, 3, 3, border_mode='same', init=init)(pool4)
        if bn:
            conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv5 = Convolution2D(conv_channel * 16, 3, 3, border_mode='same', init=init)(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(conv_channel * 8, 3, 3, border_mode='same', init=init)(up6)
        if bn:
            conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        conv6 = Convolution2D(conv_channel * 8, 3, 3, border_mode='same', init=init)(conv6)
        if bn:
            conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    else:
        up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(conv_channel*4, 3, 3, border_mode='same', init=init)(up7)
    if bn:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Convolution2D(conv_channel*4, 3, 3, border_mode='same', init=init)(conv7)
    if bn:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(conv_channel*2, 3, 3, border_mode='same', init=init)(up8)
    if bn:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv8 = Convolution2D(conv_channel*2, 3, 3, border_mode='same', init=init)(conv8)
    if bn:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(conv_channel, 3, 3, border_mode='same', init=init)(up9)
    if bn:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv9 = Convolution2D(conv_channel, 3, 3, border_mode='same', init=init)(conv9)
    if bn:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)

    # Output Convolution Layer with 10 binary classes
    outmap = Convolution2D(N_Cls, 1, 1, activation='sigmoid')(conv9)
    if use_sample_weights:
        outmap = Reshape((N_Cls, 160*160))(outmap)
        outmap = Permute((2,1))(outmap)

    model = Model(input=inputs, output=outmap)

    if use_sample_weights:
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'],
                      sample_weight_mode="temporal")
    else:
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calc_jacc(model, logger, dims, visual_name, img, msk, use_sample_weights, N_Cls=10):
    """
    Finds the optimal thresholds for the metric to be maximized: Average Jaccard Index over areas with positive mask
    over all 10 classes.
    """
    ind_scores, trs, trs_bin = [], [], []

    prd = model.predict(img, batch_size=16)
    if use_sample_weights:
        # Output is (None, 160*160, 10), reshape to (None, 10, 160, 160)
        prd = np.rollaxis(prd, 2, 1)
        prd = prd.reshape(prd.shape[0], N_Cls, 160, 160)
        msk = np.rollaxis(msk, 2, 1)
        msk = msk.reshape(msk.shape[0], N_Cls, 160, 160)
    print(prd.shape, msk.shape)

    def compute_jaccard(threshold, t_prd, t_msk):
        pred_binary_ys = t_prd >= threshold
        tp, fp, fn = ((pred_binary_ys & t_msk).sum(),
                      (pred_binary_ys & ~t_msk).sum(),
                      (~pred_binary_ys & t_msk).sum())
        jaccard = tp / (tp + fp + fn)
        return jaccard

    for i in range(N_Cls):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.flatten()
        t_msk = t_msk == 1
        t_prd = t_prd.flatten()

        best_jac, best_thresh = 0, 0
        for k in [.01, .02, .03, .04, .05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .45,
              .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]:
            tr = k
            jk = compute_jaccard(tr, t_prd, t_msk)
            if jk > best_jac:
                best_jac = jk
                best_thresh = tr
        print("{}: Max Jaccard von {:.4f} bei >= {:.4f}".format(class_list[i], best_jac, best_thresh))
        logger.info("{}: Max Jaccard von {:.4f} bei >= {:.4f}".format(class_list[i], best_jac, best_thresh))
        # Liste von average Jaccard Scores über alle Validation Crops
        ind_scores.append(best_jac)
        # Liste der besten Thresholds
        trs.append(best_thresh)
    avg_score = sum(ind_scores) / 10.0
    np.save("data/thresholds_unet_{}_{:.4f}".format(visual_name, avg_score), trs)
    print("Average Jaccard: {:.4f}".format(avg_score))
    logger.info("Average Jaccard: {:.4f}".format(avg_score))
    return avg_score, trs, ind_scores

def visualize_training(loss_train, loss_eval, name, acc_train, acc_eval):
    """
    Visualizes training with log_loss, loss and accuracy plot over training and evaluation sets.
    """
    plt.semilogy(loss_train, basey=2)
    plt.semilogy(loss_eval, basey=2, c="red")
    plt.title('{} model loss'.format(name))
    plt.ylabel('loss')
    plt.xlabel('batch')
    plt.legend(['train', 'eval'], loc='upper left')
    plt.savefig("model_selection/log_loss_{}.png".format(name), bbox_inches="tight", pad_inches=1)
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(loss_train)
    plt.plot(loss_eval, c="red")
    plt.title('{} model loss'.format(name))
    plt.ylabel('loss')
    plt.xlabel('batch')
    plt.legend(['train', 'eval'], loc='upper left')
    plt.savefig("model_selection/loss_{}.png".format(name), bbox_inches="tight", pad_inches=1)
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(acc_train)
    plt.plot(acc_eval, c="red")
    plt.title('{} model accuracy'.format(name))
    plt.ylabel('accuracy')
    plt.xlabel('batch')
    plt.ylim([0.9, 1])
    plt.legend(['train', 'eval'], loc='lower right')
    os.makedirs("../plots", exist_ok=True)
    plt.savefig("../plots/acc_{}.png".format(name), bbox_inches="tight", pad_inches=1)
    plt.clf()
    plt.cla()
    plt.close()

def visualize_scores(avg_scores, ind_scores_over_time, trs, name):
    """
    Visualizes the validation Jaccard Scores for all ten classes over the epochs.
    """
    plt.plot(avg_scores, lw=3)
    for z in range(10):
        plt.plot(ind_scores_over_time[z], ls="--")
    plt.title('Jaccard Scores')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    legend = plt.legend(["Avg Score", "Buildings", "Structures", "Road", "Track", "Trees", "Crops", "Waterway",
                  "Standing Water", "Trucks", "Cars"], loc='upper left', frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    os.makedirs("../plots", exist_ok=True)
    plt.savefig("../plots/scores_{}.png".format(name), bbox_inches="tight", pad_inches=1)
    plt.clf()
    plt.cla()
    plt.close()

def train_net(logger, dims=20, deep=True, conv_channel=32, init="glorot_uniform", fast=False, num_iterations=20,
              visual_name="", lr_start=1e-3, LR_decay=0.95, size=1600, input_name="new_eval", N_Cls=10,
              bn=True, batch_size=32, input=None, use_sample_weights=False, mins=None, maxs=None):
    """
    Trains a U-Net simultaneously for all ten classes.
    """
    print("start train net")
    model = get_unet(lr=lr_start, deep=deep, dims=dims, conv_channel=conv_channel, bn=bn,
                     use_sample_weights=use_sample_weights, init=init, N_Cls=N_Cls)
    if input is not None:
        model.load_weights('weights/{}'.format(input))
        print("Loaded {}".format(input))
        logger.info("Loaded {}".format(input))
    logger.info("Channel: {}".format(dims))
    logger.info("Inputsize: {}".format(size))
    logger.info("conv_channel: {}".format(conv_channel))
    if bn:
        logger.info("Batch Normalization: YES")
    else:
        logger.info("Batch Normalization: NO")
    logger.info("Batchsize: {}".format(batch_size))
    logger.info("Start LR: {}".format(lr_start))
    logger.info("Multiplicative LR Decay: {}".format(LR_decay))
    print("Model has {} Parameters".format(model.count_params()))
    logger.info("Model has {} Parameters".format(model.count_params()))

    x_val = np.load('../data/x_eval_{}.npy'.format(input_name))
    y_val = np.load('../data/y_eval_{}.npy'.format(input_name))

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    print("-------------------------------------------------------------------------------------")
    logger.info("-------------------------------------------------------------------------------------")
    for z in range(N_Cls):
        print("{:.4f}% Class {} in eval set".format(
            100 * y_val[:,z].sum() / (y_val.shape[0] * y_val.shape[2] * y_val.shape[3]), class_list[z]))
        logger.info("{:.4f}% Class {} in eval set".format(
            100 * y_val[:,z].sum() / (y_val.shape[0] * y_val.shape[2] * y_val.shape[3]), class_list[z]))
    if fast:
        x_val, y_val = unison_shuffled_copies(x_val, y_val)
        x_trn = x_val[:200]
        y_trn = y_val[:200]
        x_val = x_val[200:]
        y_val = y_val[200:]
    else:
        x_trn = np.load('data/x_trn_{}.npy'.format(input_name))
        y_trn = np.load('data/y_trn_{}.npy'.format(input_name))
        x_trn, y_trn = unison_shuffled_copies(x_trn, y_trn)
        for z in range(N_Cls):
            print("{:.4f}% Class {} in training set".format(100*y_trn[:,z].sum()/(y_trn.shape[0]*160*160),
                                                            class_list[z]))
            logger.info("{:.4f}% Class {} in training set".format(100*y_trn[:,z].sum()/(y_trn.shape[0]*160*160),
                                                                  class_list[z]))

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accs = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accs.append(logs.get('acc'))

    history = LossHistory()
    avg_scores, trs = [], []
    ind_scores_over_time = {}
    for z in range(N_Cls):
        ind_scores_over_time[z] = []
    loss_train = np.zeros(0)
    acc_train = np.zeros(0)
    loss_eval = np.zeros(0)
    acc_eval = np.zeros(0)
    loss_eval_once = np.zeros(0)

    def min_max_normalize(bands, mins, maxs):
        out = np.zeros_like(bands).astype(np.float32)
        n = bands.shape[1]
        for i in range(n):
            a = 0
            b = 1
            c = mins[i]
            d = maxs[i]
            t = a + (bands[:, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, i] = t
        return out.astype(np.float32)

    # Normalization: Scale with Min/Max
    x_trn = min_max_normalize(x_trn, mins, maxs)
    x_val = min_max_normalize(x_val, mins, maxs)
    # Center to zero mean and save means for later use
    means = x_trn.mean(axis=(0,2,3))
    np.save("../data/means_all_{}".format(visual_name), means)
    for k in range(dims):
        x_trn[:,k] -= means[k]
        x_val[:,k] -= means[k]
        print(x_trn[:,k].mean())
    os.makedirs("../weights", exist_ok=True)
    model_checkpoint = ModelCheckpoint('../weights/unet_10classes.hdf5', monitor='loss', save_best_only=True)
    if use_sample_weights:
        y_trn = np.transpose(y_trn, (0,2,3,1))
        y_val = np.transpose(y_val, (0,2,3,1))
        y_trn = y_trn.reshape((y_trn.shape[0], y_trn.shape[1]*y_trn.shape[2], N_Cls))
        y_val = y_val.reshape((y_val.shape[0], y_val.shape[1]*y_val.shape[2], N_Cls))
        count_classes = []
        sum_unequal_zero = 0
        for j in range(N_Cls):
            count_non = np.count_nonzero(y_trn[:,:,j])
            count_classes.append(count_non)
            sum_unequal_zero += count_non
        count_zeros = y_trn.shape[0]*y_trn.shape[1]*N_Cls - sum_unequal_zero
        count_classes = [count_zeros*1.0/val for val in count_classes]
        sample_weights = np.ones((y_trn.shape[0], y_trn.shape[1]))
        for j in range(N_Cls):
            sample_weights[y_trn[:,:,j] == 1] = count_classes[j]**0.5
            print("{} has weight {}".format(class_list[j], count_classes[j]**0.5))
            logger.info("{} has weight {}".format(class_list[j], count_classes[j]** 0.5))

    for iteration in range(num_iterations):
        print("ITERATION: {}".format(iteration))
        logger.info("ITERATION: {}".format(iteration))
        start = time.time()
        if LR_decay:
            if iteration > 0:
                # multiplicative learning rate decay
                new_LR = float(model.optimizer.lr.get_value() * LR_decay)
                model.optimizer.lr.set_value(new_LR)
        print("LR: {}".format(model.optimizer.lr.get_value()))
        print("-------------------------------------------------------------------------------------")
        logger.info("LR: {}".format(model.optimizer.lr.get_value()))
        logger.info("-------------------------------------------------------------------------------------")
        if use_sample_weights:
            model.fit(x_trn, y_trn, batch_size=batch_size, nb_epoch=1, verbose=1,
                      shuffle=True, validation_data=(x_val, y_val), callbacks=[history, model_checkpoint],
                      sample_weight=sample_weights)
        else:
            model.fit(x_trn, y_trn, batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True,
                      validation_data=(x_val, y_val), callbacks=[history, model_checkpoint])
        loss_train = np.concatenate([loss_train, np.stack([j for j in history.losses])])
        acc_train = np.concatenate([acc_train, np.stack([j for j in history.accs])])
        for metric in ["acc", "loss", "val_acc", "val_loss"]:
            logger.info("{}: {}".format(metric, model.history.history[metric]))
        if fast:
            batches = len(np.stack([j for j in history.losses]))
        else:
            batches = len(np.stack([j for j in history.losses]))
        for l in range(batches):
            loss_eval = np.append(loss_eval, model.history.history["val_loss"])
            acc_eval = np.append(acc_eval, model.history.history["val_acc"])
        loss_eval_once = np.append(loss_eval_once, model.history.history["val_loss"])
        # Calculate best score and thresholds
        avg_score, trs, ind_scores = calc_jacc(model, logger, dims, visual_name, x_val, y_val,
                                               use_sample_weights, N_Cls=N_Cls)
        avg_scores.append(avg_score)
        for z in range(N_Cls):
            ind_scores_over_time[z].append(ind_scores[z])
        model.save_weights('../weights/unet_{}_{:.4f}'.format(visual_name, avg_score))

        visualize_training(loss_train, loss_eval, name="{}".format(visual_name), acc_train=acc_train,
                           acc_eval=acc_eval)
        visualize_scores(avg_scores, ind_scores_over_time, trs, name="{}".format(visual_name))
        print("Iteration {} took {:.2f}s.".format(iteration, time.time() - start))
    return model, avg_score, trs, avg_scores

if __name__ == "__main__":
    logger = init_logging("../logs/{}.log".format(datetime.now().strftme("%d-%m-%y")),
                          "START: Training")
    # precomputed minimum and maximum values for all spectral bands
    mins = [55.0, 167.0, 99.0, 174.0, 182.0, 144.0, 158.0, 132.0, 61.0, 138.0, 160.0, 113.0, 672.0, 490.0, 435.0,
            391.0, 55.0, 168.0, 187.0, 55.0]
    maxs = [2047.0, 2047.0, 2047.0, 2040.0, 2035.0, 2047.0, 2047.0, 2047.0, 2047.0, 2047.0, 2047.0, 2047.0, 15410.0,
            16050.0, 16255.0, 16008.0, 15933.0, 15805.0, 15878.0, 15746.0]

    model, avg_score, trs, avg_scores = train_net(
        logger=logger, deep=True, conv_channel=32, dims=20, fast=False, num_iterations=40, bn=False,
        visual_name="conv32_nobn_nodo_bs16_decay.97", lr_start=1e-3, LR_decay=0.97,
        input_name="all_1600_denom.5_20bands", batch_size=16, size=1600, use_sample_weights=True,
        mins=mins, maxs=maxs, init="glorot_uniform")

