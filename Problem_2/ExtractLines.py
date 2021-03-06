#!/usr/bin/env python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *
import pdb

############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-splitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split.
        endIdx: ending index of segment to be split.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here ##########
    temp_alpha, temp_r = FitLine(theta[startIdx:endIdx], rho[startIdx:endIdx])

    if (endIdx - startIdx) <= params['MIN_POINTS_PER_SEGMENT'] :
        return np.array([temp_alpha]), np.array([temp_r]), np.array([[startIdx, endIdx]])

    s = FindSplit(theta[startIdx:endIdx], rho[startIdx:endIdx], temp_alpha, temp_r, params)
    if s == -1:
        return np.array([temp_alpha]), np.array([temp_r]), np.array([[startIdx, endIdx]])

    alpha1, r1, idx1 = SplitLinesRecursive(theta, rho, startIdx, startIdx + s, params)
    alpha2, r2, idx2 = SplitLinesRecursive(theta, rho, startIdx + s, endIdx, params)

    alpha = np.concatenate((alpha1, alpha2)) 
    r = np.concatenate((r1, r2))
    idx = np.concatenate((idx1, idx2))
    ########## Code ends here ##########
    return alpha, r, idx

def FindSplit(theta, rho, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).
    '''
    ########## Code starts here ##########
    d = np.absolute(rho*np.cos((theta-alpha))-r) #create array of distances of each point to line segment
    while True:
        max_dist_index = np.argmax(d)
        max_distance = max(d)
        if max_distance == 0:
            splitIdx = -1
            break
        min_segment_length = min(np.size(d[0:max_dist_index]), np.size(d[max_dist_index:]))
        if max_distance >= params['LINE_POINT_DIST_THRESHOLD'] and min_segment_length >= params['MIN_POINTS_PER_SEGMENT']:
            splitIdx = max_dist_index
            break
        
        d[max_dist_index] = 0 # change to zero to move to next largest value
    ########## Code ends here ##########
    return splitIdx

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads).
        r: 'r' of best fit for range data (1 number) (m).
    '''
    ########## Code starts here ##########
    n = theta.size
    # for alpha:
    rho_sqd = rho**2
    sin_twotheta = np.sin((2*theta))
    cos_twotheta = np.cos((2*theta))

    first_term = np.sum(np.multiply(rho_sqd, sin_twotheta))
    second_term = (2.0/n)*np.sum(rho[i]*rho[j]*np.cos(theta[i])*np.sin(theta[j]) for i in range(n) for j in range(n))
    third_term = np.sum(np.multiply(rho_sqd, cos_twotheta))
    fourth_term = (1.0/n)*np.sum(rho[i]*rho[j]*np.cos((theta[i]+theta[j])) for i in range(n) for j in range(n))

    alpha = .5*np.arctan2((first_term - second_term),(third_term - fourth_term)) + (np.pi/2)

    # for r
    r = (1.0/n)*np.sum(np.multiply(rho,np.cos((theta-alpha))))
    ########## Code ends here ##########
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########
    last_num_segs = 0
    alphaOut = alpha
    rOut = r
    pointIdxOut = pointIdx

    while True:
        num_segs = rOut.size
        if num_segs == last_num_segs:
            break
        for curr_seg_idx in range((num_segs-1)):
            next_seg_idx = curr_seg_idx + 1
            start = pointIdxOut[curr_seg_idx][0]
            end = pointIdxOut[next_seg_idx][1]
            new_theta = theta[start:end]
            new_rho = rho[start:end]
            new_alpha, new_r = FitLine(new_theta, new_rho)

            splitIdx = FindSplit(new_theta, new_rho, new_alpha, new_r, params)
            if splitIdx == -1:
                # adjust line segment collection
                alphaOut[curr_seg_idx] = new_alpha
                rOut[curr_seg_idx] = new_r
                pointIdxOut[curr_seg_idx,:] = np.array([[start, end]])
                # remove the neighbor that has been merged
                alphaOut = np.delete(alphaOut, next_seg_idx)
                rOut = np.delete(rOut, next_seg_idx)
                pointIdxOut = np.delete(pointIdxOut, next_seg_idx,0)
                break

        last_num_segs = num_segs
    
    ########## Code ends here ##########
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.002 #0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = .06#0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2#4  # minimum number of points per line segment
    MAX_P2P_DIST = .3#1.0  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    # filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
