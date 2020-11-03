import numpy as np
import scipy.linalg    # you may find scipy.linalg.block_diag useful
import turtlebot_model as tb

class Ekf(object):
    """
    Base class for EKF Localization and SLAM.

    Usage:
        ekf = EKF(x0, Sigma0, R)
        while True:
            ekf.transition_update(u, dt)
            ekf.measurement_update(z, Q)
            localized_state = ekf.x
    """

    def __init__(self, x0, Sigma0, R):
        """
        EKF constructor.

        Inputs:
                x0: np.array[n,]  - initial belief mean.
            Sigma0: np.array[n,n] - initial belief covariance.
                 R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.x = x0          # Gaussian belief mean
        self.Sigma = Sigma0  # Gaussian belief covariance
        self.R = R           # Control noise covariance (corresponding to dt = 1 second)

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating (self.x, self.Sigma).

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        g, Gx, Gu = self.transition_model(u, dt)

        ########## Code starts here ##########
        # Just follow equation as given in part (ii) of pset
        sig, R = self.Sigma, self.R
        self.x = g
        self.Sigma = np.matmul(np.matmul(Gx, sig), Gx.T) + dt * np.matmul(np.matmul(Gu, R), Gu.T)
        ########## Code ends here ##########

    def transition_model(self, u, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Outputs:
             g: np.array[n,]  - result of belief mean propagated according to the
                                system dynamics with control u for dt seconds.
            Gx: np.array[n,n] - Jacobian of g with respect to belief mean self.x.
            Gu: np.array[n,2] - Jacobian of g with respect to control u.
        """
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        z, Q, H = self.measurement_model(z_raw, Q_raw)
        if z is None:
            # Don't update if measurement is invalid
            # (e.g., no line matches for line-based EKF localization)
            return

        ########## Code starts here ##########
        Sig = self.Sigma
        x = self.x

        S     = np.matmul(np.matmul(H, Sig), H.T) + Q
        S_inv = np.linalg.inv(S)
        K     = np.matmul(np.matmul(Sig, H.T), S_inv)
        self.x     = x + np.matmul(K, z)
        self.Sigma = Sig - np.matmul(np.matmul(K, S), K.T)
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction). Also returns the associated Jacobian for EKF
        linearization.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2K,]   - measurement mean.
            Q: np.array[2K,2K] - measurement covariance.
            H: np.array[2K,n]  - Jacobian of z with respect to the belief mean self.x.
        """
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class EkfLocalization(Ekf):
    """
    EKF Localization.
    """

    def __init__(self, x0, Sigma0, R, map_lines, tf_base_to_camera, g):
        """
        EkfLocalization constructor.

        Inputs:
                       x0: np.array[3,]  - initial belief mean.
                   Sigma0: np.array[3,3] - initial belief covariance.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Turtlebot dynamics (unicycle model).
        """

        ########## Code starts here ##########
        # TODO: Compute g, Gx, Gu using tb.compute_dynamics().
        g, Gx, Gu = tb.compute_dynamics(self.x, u, dt)
        ########## Code ends here ##########

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature.
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print("Scanner sees {} lines but can't associate them with any map entries."
                  .format(z_raw.shape[1]))
            return None, None, None

        ########## Code starts here ##########
        # Just follow equations in pset (v)
        v = np.array(v_list)
        # Q = np.array(Q_list)
        H = np.array(H_list)
        d = H.shape[2]
        
        z = v.reshape(-1)
        Q = scipy.linalg.block_diag(*Q_list)
        # Q = scipy.sparse.block_diag(Q_list) # This creates an nd.array matrix as it's sparse.
        #                                     # Creates problems downstream.
        H = H.reshape(-1, d)
        ########## Code ends here ##########

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            v_list: [np.array[2,]]  - list of at most I innovation vectors
                                      (predicted map measurement - scanner measurement).
            Q_list: [np.array[2,2]] - list of covariance matrices of the
                                      innovation vectors (from scanner uncertainty).
            H_list: [np.array[2,3]] - list of Jacobians of the innovation
                                      vectors with respect to the belief mean self.x.
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        hs, Hs = self.compute_predicted_measurements()

        ########## Code starts here ##########

        # Vectorized version
        
        # DEVNOTE: Since the matrices involved are small, ensuring array contiguity
        # before matmul doesn't make much difference.

        # Conversions
        d       = self.x.shape[0]       # Should be 3 for (x, y, th)
        Sig     = self.Sigma            # shape(d, d). Belief (Gaussian) covariance 
        Q_raw   = np.asarray(Q_raw)     # shape(n_mea, 2, 2)
        Hs      = np.asarray(Hs)        # shape(n_lin, 2, d)
        hs      = hs.T                  # shape(n_lin, 2)
        z_raw   = z_raw.T               # shape(n_mea, 2)
        # Dimension 2 is the minimal parameterization of a straight line.

        n_mea  = z_raw.shape[0]      # Num of measured lines
        n_lin  = Hs.shape[0]         # Num of known lines on map
        thresh = self.g * self.g     # Association threshold as given by pset

        v_mat = z_raw[None, :, :] - hs[:, None, :] # shape(n_lin, n_mea, 2)

        S_mat = np.matmul(Hs, Sig)                            # shape(n_lin, 2, d)
        S_mat = np.matmul(S_mat, np.transpose(Hs, (0, 2, 1))) # shape(n_lin, 2, 2)
        S_mat = S_mat[:, None, :, :] + Q_raw[None, :, :, :]   # shape(n_lin, n_mea, 2, 2)

        Sinv_mat = np.linalg.inv(S_mat)                         # shape(n_lin, n_mea, 2, 2)
        v_fat  = v_mat[..., None]                               # shape(n_lin, n_mea, 2, 1)
        d_mat  = np.matmul(np.transpose(v_fat, (0, 1, 3, 2)), Sinv_mat)
        d_mat  = np.matmul(d_mat, v_fat)                        # shape(n_lin, n_mea, 1, 1)
        d_mat  = np.reshape(d_mat, (n_lin, n_mea))              # shape(n_lin, n_mea)

        # Now we have a matrix of Mahalanobis distances between every measurement and
        # every known line. Only thing left to do is to grab the right indices.

        idx = np.linspace(0, n_mea, n_mea, endpoint=False, dtype=np.int)   # just counts up in [0, n_mea-1]
        d_argmin = np.argmin(d_mat, axis=0) # shape(n_mea)
        d_min = d_mat[d_argmin, idx]        # shape(n_mea)

        mea_idxs = idx[d_min < thresh]
        lin_idxs = d_argmin[d_min < thresh]

        v_list = v_mat[lin_idxs, mea_idxs, :].tolist()  # shape(<=n_mea, 2)
        Q_list = Q_raw[mea_idxs, :, :].tolist()         # shape(<=n_mea, 2, 2)
        H_list = Hs[lin_idxs, :, :].tolist()            # shape(<=n_mea, 2, d)

        """
        # Naive loopy version.
        d   = self.x.shape[0] # Should be 3 for (x, y, th)
        Sig = self.Sigma      # shape(d, d). Belief (Gaussian) covariance 
        Hs = np.asarray(Hs)   # shape(n_lin, 2, d)
        # hs = hs             # shape(2, n_lin)
        n_mea = z_raw.shape[1]      # Num of measured lines
        n_lin = Hs.shape[0]         # Num of known lines on map
        thresh = self.g * self.g    # Association threshold as given by pset

        v_list = []
        Q_list = []
        H_list = []

        for i in range(n_mea): # For each measurement we have
            zi = z_raw[:, i] # observation
            Qi = Q_raw[i] # shape(2,2) observation covariance

            d_min = np.inf

            v, Q, H = None, None, None
            for j in range(n_lin): # Compare against each known line
                hj = hs[:, j]
                Hj = Hs[j]

                vij = zi - hj
                Sij = np.matmul(np.matmul(Hj, Sig), Hj.T) + Qi
                dij = np.matmul(np.matmul(vij.T, np.linalg.inv(Sij)), vij)

                if dij < d_min:
                    d_min = dij
                    v, Q, H = vij, Qi, Hj

            if d_min < thresh:
                v_list.append(v)
                Q_list.append(Q)
                H_list.append(H)
        """

        ########## Code ends here ##########

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Outputs:
                 hs: np.array[2,J]  - J line parameters in the scanner (camera) frame.
            Hx_list: [np.array[2,3]] - list of Jacobians of h with respect to the belief mean self.x.
        """
        hs = np.zeros_like(self.map_lines)
        Hx_list = []
        for j in range(self.map_lines.shape[1]):
            ########## Code starts here ##########
            line = self.map_lines[:,j]
            x = self.x
            tf_base_to_camera = self.tf_base_to_camera
            h, Hx = tb.transform_line_to_scanner_frame(line, x, tf_base_to_camera)
            ########## Code ends here ##########

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[:,j] = h
            Hx_list.append(Hx)

        return hs, Hx_list


class EkfSlam(Ekf):
    """
    EKF SLAM.
    """

    def __init__(self, x0, Sigma0, R, tf_base_to_camera, g):
        """
        EKFSLAM constructor.

        Inputs:
                       x0: np.array[3+2J,]     - initial belief mean.
                   Sigma0: np.array[3+2J,3+2J] - initial belief covariance.
                        R: np.array[2,2]       - control noise covariance
                                                 (corresponding to dt = 1 second).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Combined Turtlebot + map dynamics.
        Adapt this method from EkfLocalization.transition_model().
        """
        g = np.copy(self.x)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size, 2))

        ########## Code starts here ##########
        g_tb, Gx_tb, Gu_tb = tb.compute_dynamics(self.x[:3], u, dt)
        g[0:3] = g_tb
        Gx[0:3, 0:3] = Gx_tb
        Gu[0:3, 0:2] = Gu_tb
        ########## Code ends here ##########

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Combined Turtlebot + map measurement model.
        Adapt this method from EkfLocalization.measurement_model().
        
        The ingredients for this model should look very similar to those for
        EkfLocalization. In particular, essentially the only thing that needs to
        change is the computation of Hx in self.compute_predicted_measurements()
        and how that method is called in self.compute_innovations() (i.e.,
        instead of getting world-frame line parameters from self.map_lines, you
        must extract them from the state self.x).
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print("Scanner sees {} lines but can't associate them with any map entries."
                  .format(z_raw.shape[1]))
            return None, None, None

        ########## Code starts here ##########
        # TODO: Compute z, Q, H.
        # Hint: Should be identical to EkfLocalization.measurement_model().
        v = np.array(v_list)
        H = np.array(H_list)
        d = H.shape[2]
        
        z = v.reshape(-1)
        Q = scipy.linalg.block_diag(*Q_list)
        H = H.reshape(-1, d)
        ########## Code ends here ##########

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Adapt this method from EkfLocalization.compute_innovations().
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        hs, Hs = self.compute_predicted_measurements()

        ########## Code starts here ##########

        # No change whatsoever.
        
        # Vectorized version
        
        # DEVNOTE: Since the matrices involved are small, ensuring array contiguity
        # before matmul doesn't make much difference.

        # Conversions
        d       = self.x.shape[0]       # Should be 3 for (x, y, th)
        Sig     = self.Sigma            # shape(d, d). Belief (Gaussian) covariance 
        Q_raw   = np.asarray(Q_raw)     # shape(n_mea, 2, 2)
        Hs      = np.asarray(Hs)        # shape(n_lin, 2, d)
        hs      = hs.T                  # shape(n_lin, 2)
        z_raw   = z_raw.T               # shape(n_mea, 2)
        # Dimension 2 is the minimal parameterization of a straight line.

        n_mea  = z_raw.shape[0]      # Num of measured lines
        n_lin  = Hs.shape[0]         # Num of known lines on map
        thresh = self.g * self.g     # Association threshold as given by pset

        # We have to ensure the angle is in range [0, pi] so we can't just subtract directly.
        # Vectorize and apply the angle_diff() function given
        # v_mat = z_raw[None, :, :] - hs[:, None, :] # shape(n_lin, n_mea, 2)

        z_mat = z_raw[None, :, :]   # shape(n_lin, n_mea, 2)
        h_mat = hs[:, None, :]      # shape(n_lin, n_mea, 2)
        # Compute angle diff
        z_alp, h_alp = z_mat[:, :, 0], h_mat[:, :, 0] # shape(n_lin, n_mea)
        z_alp, h_alp = z_alp % (2.*np.pi), h_alp % (2.*np.pi)
        diff = z_alp - h_alp
        idx = np.abs(diff) > np.pi
        sign = 2. * (diff[idx] < 0.) - 1.
        diff[idx] += sign * 2. * np.pi
        v_alp = diff
        # Reconstruct v
        v_r = z_mat[:, :, 1] - h_mat[:, :, 1]
        v_mat = np.stack((v_alp, v_r), axis=2) # shape(n_lin, n_mea, 2)

        S_mat = np.matmul(Hs, Sig)                            # shape(n_lin, 2, d)
        S_mat = np.matmul(S_mat, np.transpose(Hs, (0, 2, 1))) # shape(n_lin, 2, 2)
        S_mat = S_mat[:, None, :, :] + Q_raw[None, :, :, :]   # shape(n_lin, n_mea, 2, 2)

        Sinv_mat = np.linalg.inv(S_mat)                         # shape(n_lin, n_mea, 2, 2)
        v_fat  = v_mat[..., None]                               # shape(n_lin, n_mea, 2, 1)
        d_mat  = np.matmul(np.transpose(v_fat, (0, 1, 3, 2)), Sinv_mat)
        d_mat  = np.matmul(d_mat, v_fat)                        # shape(n_lin, n_mea, 1, 1)
        d_mat  = np.reshape(d_mat, (n_lin, n_mea))              # shape(n_lin, n_mea)

        # Now we have a matrix of Mahalanobis distances between every measurement and
        # every known line. Only thing left to do is to grab the right indices.

        idx = np.linspace(0, n_mea, n_mea, endpoint=False, dtype=np.int)   # just counts up in [0, n_mea-1]
        d_argmin = np.argmin(d_mat, axis=0) # shape(n_mea)
        d_min = d_mat[d_argmin, idx]        # shape(n_mea)

        mea_idxs = idx[d_min < thresh]
        lin_idxs = d_argmin[d_min < thresh]

        v_list = v_mat[lin_idxs, mea_idxs, :].tolist()  # shape(<=n_mea, 2)
        Q_list = Q_raw[mea_idxs, :, :].tolist()         # shape(<=n_mea, 2, 2)
        H_list = Hs[lin_idxs, :, :].tolist()            # shape(<=n_mea, 2, d)

        ########## Code ends here ##########

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Adapt this method from EkfLocalization.compute_predicted_measurements().
        """
        J = (self.x.size - 3) // 2
        hs = np.zeros((2, J))
        Hx_list = []
        X = self.x[:3]
        for j in range(J):
            idx_j = 3 + 2 * j
            alpha, r = self.x[idx_j:idx_j+2]

            Hx = np.zeros((2,self.x.size))

            ########## Code starts here ##########
            # TODO: Compute h, Hx.

            # Exactly the same as tb.transform_line_to_scanner_frame()
            alp = alpha
            x, y, th = X
            xcam_R, ycam_R, thcam_R = self.tf_base_to_camera   # Camera pose. in Robot frame.
            C_Rh = np.array([xcam_R, ycam_R, 1])          # Camera pose in homogeneous coords. Robot frame.

            RW_T = np.array([
                [np.cos(th), -np.sin(th), x],
                [np.sin(th),  np.cos(th), y],
                [0, 0, 1]
            ])

            Cam_h = np.matmul(RW_T, C_Rh) # Camera in homogeneous coords. World frame.
            xcam, ycam, hcam = Cam_h
            xcam, ycam = xcam/hcam, ycam/hcam  # Camera in World frame coords.

            alp_C = alp - th - thcam_R
            r_C = r - xcam*np.cos(alp) - ycam*np.sin(alp)
            h = np.array([alp_C, r_C])

            dalpC_dxR  = 0
            dalpC_dyR  = 0
            dalpC_dthR = -1

            drC_dxW = -np.cos(alp)
            drC_dyW = -np.sin(alp)
            drC_dthW = (-np.cos(alp) * (-xcam_R*np.sin(th) - ycam_R*np.cos(th))
                        -np.sin(alp) * ( xcam_R*np.cos(th) - ycam_R*np.sin(th))
                        )

            # Hx is bigger now
            Hx[0:2,0:3] = np.array([
                [dalpC_dxR, dalpC_dyR, dalpC_dthR],
                [drC_dxW,   drC_dyW,   drC_dthW]
            ])
            # So the columns of Hx go dx, dy, dth, dalp1, dr1, dalp2, dr2, ...

            # First two map lines are assumed fixed so we don't want to propagate
            # any measurement correction to them. i.e. Entries all zero.
            # Otherwise:

            # dalpC_dalpC = 1
            # dalpC_drC   = 0
            drC_dalpC     = xcam*np.sin(alp) - ycam*np.cos(alp)
            # drC_drC     = 1

            if j >= 2:
                Hx[:,idx_j:idx_j+2] = np.eye(2)
                Hx[1,idx_j] = drC_dalpC
            ########## Code ends here ##########

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[:,j] = h
            Hx_list.append(Hx)

        return hs, Hx_list
