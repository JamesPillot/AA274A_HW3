#!/usr/bin/env python

import pdb
import os
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns

class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]

    def genCornerCoordinates(self, u_meas, v_meas):
        '''
        Inputs:
            u_meas: a list of arrays where each array are the u values for each board.
            v_meas: a list of arrays where each array are the v values for each board.
        Output:
            corner_coordinates: a tuple (Xg, Yg) where Xg/Yg is a list of arrays where each array are the x/y values for each board.

        HINT: u_meas, v_meas starts at the blue end, and finishes with the pink end
        HINT: our solution does not use the u_meas and v_meas values
        HINT: it does not matter where your frame it, as long as you are consistent!
        '''
        ########## Code starts here ##########
        # Board dimensions
        N_cols = 9
        M_rows = 7

        # Xg and Yg will be list of np arrays 
        Xg = []
        Yg = []

        # Go through each board
        for curr_board_u in u_meas:
            x = []
            y = []
            
            # Populate x and y arrays (created as lists and then converted later)
            for index in range(np.size(curr_board_u)):
                xval = (index % N_cols) * self.d_square
                x.append(xval)
                yval = np.floor((index / N_cols)) * self.d_square
                y.append(yval)
	
            # Update Xg and Yg 
            Xg.append(np.array(x))
            Yg.append(np.array(y))

        corner_coordinates = (Xg, Yg)
        ########## Code ends here ##########
        return corner_coordinates

    def estimateHomography(self, u_meas, v_meas, X, Y):    # Zhang Appendix A
        '''
        Inputs:
            u_meas: an array of the u values for a board.
            v_meas: an array of the v values for a board.
            X: an array of the X values for a board. (from genCornerCoordinates)
            Y: an array of the Y values for a board. (from genCornerCoordinates)
        Output:
            H: the homography matrix. its size is 3x3

        HINT: What is the size of the matrix L?
        HINT: What are the outputs of the np.linalg.svd function? Based on this, where does the eigenvector corresponding to the smallest eigen value live?
        HINT: np.stack and/or np.hstack may come in handy here.
        '''
        ########## Code starts here ##########

        # Start with building P_tilde
        sizeX  = np.size(X) # dimension of X "dim(X)" which is also dim(Y) = dim(u_meas) = dim(v_meas)

        onesvec = np.ones(sizeX) # vector of ones for homogeneus representation of world coordinates
        Ph_w = np.array([X, Y, onesvec], dtype = 'float' ) 

        P_tilde_list = []
        for col in range(Ph_w.shape[1]):
            Ph_w_i = Ph_w[:,col]
	    Ph_w_i.shape = (3,1)
            Ph_w_i_t = np.transpose(Ph_w_i) # homogeneus world coordinate i transposed
            # Build block of Ptilde
            zeroesvec = np.zeros((1, np.size(Ph_w_i)))
            block = np.block([
                [Ph_w_i_t, zeroesvec, -u_meas[col]*Ph_w_i_t], # col is also current index of umeas and vmeas
                [zeroesvec, Ph_w_i_t, -v_meas[col]*Ph_w_i_t]
            ])
            P_tilde_list.append(block)

        P_tilde = np.vstack(P_tilde_list) #finally build P_tilde from list of blocks

        # Now use SVD to solve constrained least squares problem and get satisfactory m
        u, s, vh = np.linalg.svd(P_tilde)
        m = vh[-1,:] # m is first row of Vh (this gives us vh_1 which is what we want)
        m1 = m[0:3] # first column of H
        m2 = m[3:6] # second column of H
        m3 = m[6:9] # third column of H

        H = np.vstack((m1,m2,m3))
        ########## Code ends here ##########
        return H

    def getCameraIntrinsics(self, H):    # Zhang 3.1, Appendix B
        '''
        Input:
            H: a list of homography matrices for each board
        Output:
            A: the camera intrinsic matrix

        HINT: MAKE SURE YOU READ SECTION 3.1 THOROUGHLY!!! V. IMPORTANT
        HINT: What is the definition of h_ij?
        HINT: It might be cleaner to write an inner function (a function inside the getCameraIntrinsics function)
        HINT: What is the size of V?
        '''
        ########## Code starts here ##########

        # define inner function to calculate V for each Homography matrix in list H
        def calcV(i,j,currH):

	    
            first_entry = currH[0,i]*currH[0,j]
            second_entry = currH[0,i]*currH[1,j] + currH[1,i]*currH[0,j]
            third_entry = currH[1,i]*currH[1,j]
            fourth_entry = currH[2,i]*currH[0,j] + currH[0,i]*currH[2,j]
            fifth_entry = currH[2,i]*currH[1,j] + currH[1,i]*currH[2,j]
            sixth_entry = currH[2,i]*currH[2,j]

            v_ij_t = np.array([[first_entry, second_entry, third_entry, fourth_entry, fifth_entry, sixth_entry]], dtype = 'float')
            return v_ij_t
        
        V_list = []
        for curH in H:
            v_onetwo_t = calcV(0,1, curH)
            v_oneone_minus_v_twotwo_t = calcV(0,0,curH) - calcV(1,1,curH)
            cur_image_v = np.vstack((v_onetwo_t, v_oneone_minus_v_twotwo_t))
            V_list.append(cur_image_v)
        
        V = np.vstack(V_list)
        V = V.astype('float')

        # Now use SVD to solve another constrained least squares and get satisfactory b
        u, s, vh = np.linalg.svd(V)
        b = vh[-1,:] # b is last row of Vh (this gives us smallest vh_1 which is what we want)
	
        # Finally use b to back out the parameters and fill out our matrix A
        B11 = b[0]
        B12 = b[1]
        B22 = b[2]
        B13 = b[3]
        B23 = b[4]
        B33 = b[5]
        
	v0 = (B12*B13 - B11*B23)/(B11*B22 - (B12**2))
        lamb = B33 - (((B13**2) + v0*(B12*B13 - B11*B23))/(B11)) #lambda
        alpha = np.sqrt((lamb/B11))
        beta = np.sqrt((lamb*B11)/(B11*B22 - (B12**2)))
        gamma = (-B12*(alpha**2)*beta)/lamb
        u0 = ((gamma*v0)/beta) - ((B13*(alpha**2))/lamb)

        A = np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ])

        ########## Code ends here ##########
        return A

    def getExtrinsics(self, H, A):    # Zhang 3.1, Appendix C
        '''
        Inputs:
            H: a single homography matrix
            A: the camera intrinsic matrix
        Outputs:
            R: the rotation matrix
            t: the translation vector
        '''
        ########## Code starts here ##########
        Ainv = np.linalg.inv(A)
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        lamb1 = 1 / (np.linalg.norm(np.matmul(Ainv,h1))) #lambda1
        lamb2 = 1/ (np.linalg.norm(np.matmul(Ainv,h2))) #lambda2

        # Rotation matrix R
        r1 = lamb1*np.matmul(Ainv, h1)
        r2 = lamb1*np.matmul(Ainv, h2)
        r3 = np.cross(r1, r2)
        Q = np.array([r1, r2, r3]).T # build preliminary rotation matrix
        u, s, vh = np.linalg.svd(Q) # use svd to estimate the best rotation matrix
        R = np.matmul(u, vh)

        # Translation vector t
        t = lamb1*np.matmul(Ainv, h3)

        ########## Code ends here ##########
        return R, t

    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            x, y: the coordinates in the ideal normalized image plane

        '''
        ########## Code starts here ##########
        Pw = np.array([X, Y, Z], ndmin = 2)
        Pc_list = []
        for col in range(Pw.shape[1]):
            Pw_i = Pw[:,col]
            Pw_i.shape = (3,1)
            Pc_i = t + np.matmul(R,Pw_i)
            Pc_list.append(Pc_i)
        
        Pc = np.vstack(Pc_list)

	x = Pc[:,0]
        y = Pc[:,1]
 	######### Code ends here ##########
        return x, y

    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):    # Zhang 2.1, Eq. (1)
        '''
        Inputs:
            X, Y, Z: the world coordinates of the points for a given board. This is an array of 63 elements
                     X, Y come from genCornerCoordinates. Since the board is planar, we assume Z is an array of zeros.
            A: the camera intrinsic parameters
            R, t: the camera extrinsic parameters (rotation matrix and translation vector) for a given board.
        Outputs:
            u, v: the coordinates in the ideal pixel image plane
        '''
        ########## Code starts here ##########
        Ph_w = np.array([X, Y, Z, np.ones(np.size(X))], ndmin = 2) #homogeneus world coordinates

        t.shape = (3,1) #for block concatenation
        rt_block = np.block([
            [R, t]
        ])
        bigMatrix = np.matmul(A, rt_block)

        ph_list = []
        for col in range(Ph_w.shape[1]):
            Ph_w_i = Ph_w[:,col]
            Ph_w_i.shape = (4,1)
            ph_i = np.matmul(bigMatrix, Ph_w_i)
            ph_i.shape = (3,)
            ph_list.append(ph_i)
	    #pdb.set_trace()
        
        ph = np.vstack(ph_list)
        u = ph[:,0]/ph[:,2]
        v = ph[:,1]/ph[:,2]
        ########## Code ends here ##########
        return u, v

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')
            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])   # Flip Y-axis to traditional direction

        return u_meas, v_meas   # Lists of arrays (one per chessboard)
