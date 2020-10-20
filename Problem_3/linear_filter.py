#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import pdb

def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    # Build t_ij vector
    def I_vec(i,j):
        I_top = np.atleast_2d(paddedI[i,j,:])
        I_middle = np.atleast_2d(paddedI[i,(j+1):,0])
        I_bottom = np.atleast_2d(paddedI[(i+1):,j,0])
        newI = np.transpose(np.hstack([I_top, I_middle, I_bottom]))
        return newI

    # Zero padding I
    c = I.shape[2]
    n = I.shape[1]
    m = I.shape[0]
    pad_sides = int(np.floor((n/2)))
    pad_topBotom = int(np.floor((m/2)))
    paddedI = np.zeros((m+2*pad_topBotom, n+2*pad_sides, c))
    for chan in range(c):
        top_row =  np.zeros((pad_topBotom,(2*pad_sides) + n))
        mid_row = np.block([[np.zeros((m,pad_sides)), I[:,:,chan], np.zeros((m,pad_sides))]])
        bottom_row = np.zeros((pad_topBotom,(2*pad_sides) + n))
        paddedI[:,:,chan] = np.vstack((top_row, mid_row, bottom_row))

     # Build F vector
    F_top = np.atleast_2d(F[0,0,:])
    F_middle = np.atleast_2d(F[0,1:,0])
    F_bottom = np.atleast_2d(F[1:,0,0])
    newF = np.transpose(np.hstack([F_top, F_middle, F_bottom]))
    
    G = np.zeros((m,n))
    for row in range(m):
        for col in range(n):
            temp = np.dot(newF,I_vec(row,col))
            pdb.set_trace()

            # G[row, col] = 
    return G
    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    # Build t_ij vector
    def I_vec(i,j):
        I_top = np.atleast_2d(paddedI[i,j,:])
        I_middle = np.atleast_2d(paddedI[i,(j+1):,0])
        I_bottom = np.atleast_2d(paddedI[(i+1):,j,0])
        newI = np.transpose(np.hstack([I_top, I_middle, I_bottom]))
        return newI

    # Zero padding I
    c = I.shape[2]
    n = I.shape[1]
    m = I.shape[0]
    pad_sides = int(np.floor((n/2)))
    pad_topBotom = int(np.floor((m/2)))
    paddedI = np.zeros((m+2*pad_topBotom, n+2*pad_sides, c))
    for chan in range(c):
        top_row =  np.zeros((pad_topBotom,(2*pad_sides) + n))
        mid_row = np.block([[np.zeros((m,pad_sides)), I[:,:,chan], np.zeros((m,pad_sides))]])
        bottom_row = np.zeros((pad_topBotom,(2*pad_sides) + n))
        pdb.set_trace()

        paddedI[:,:,chan] = np.vstack((top_row, mid_row, bottom_row))

     # Build F vector
    F_top = np.atleast_2d(F[0,0,:])
    F_middle = np.atleast_2d(F[0,1:,0])
    F_bottom = np.atleast_2d(F[1:,0,0])
    F = np.transpose(np.hstack([F_top, F_middle, F_bottom]))
    
    G = np.zeros((m,n))
    for row in range(m):
        for col in range(n):
            G[row,col] = np.inner(F,I_vec(row,col)) / (np.linalg.norm(F)*np.linalg.norm(I_vec(row,col)))
    return G
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
