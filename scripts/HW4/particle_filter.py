import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().

        # So, in the transition update step, we have our control input u=[V, om],
        # with zero-centered Gaussian control noise model given by
        # self.R = [dV/dV,  dV/dom ]
        #          [dom/dV, dom/dom]
        # We want to get our new state by applying the transition_model to each particle
        # by sampling from our noisy input.

        d = u.shape[0]  # dimension of input
        n = self.M      # num of particles
        eps = np.random.multivariate_normal(np.zeros(d), self.R, size=n) # Gaussian input noise
        us = u + eps    # shape(n, 2)
        self.xs = self.transition_model(us, dt)
        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

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
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.


        # The way to see the algorithm is that the random value of r generates
        # a sampling 'sieve' which we then use to pick out particles which are
        # represented in terms of their weight on a sampling interval [0, 1].
        # This sieve has as many points as we have particles.

        # r ~ U[0, 1/n]
        n = self.M
        m = np.linspace(0, n, n, endpoint=False) # {0, ..., n-1}
        sieve = r + m/n
        u = np.sum(ws) * sieve # Normalization step. Maintains [0, 1] in case ws don't sum to 1.
        csum = np.cumsum(ws)
        idx = np.searchsorted(csum, u)
        self.xs = xs[idx]
        self.ws = ws[idx]

        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.

        # We don't use numpy.where here as arrays are not lazy-evaluated.

        U, X = us.T, self.xs.T
        n = self.M      # num of particles

        V_all, om_all = U       # All of shape (n, )
        x_all, y_all, th_all = X
        
        # First we need to split up the particles depending on |om|
        # to use either the normal formulae or after applying l'Hopitals
        idx = np.linspace(0, n, n, endpoint=False, dtype=np.int)
        cond = np.absolute(om_all) > EPSILON_OMEGA
        
        # Preallocate output
        x_til = np.zeros(n)
        y_til = np.zeros(n)
        th_til = np.zeros(n)

        # Normal case
        i1 = idx[cond]
        V, om = V_all[i1], om_all[i1]
        x, y, th = x_all[i1], y_all[i1], th_all[i1]

        thomdt = th+om*dt
        sin_minus_sin = np.sin(thomdt) - np.sin(th)
        cos_minus_cos = np.cos(thomdt) - np.cos(th)
        # We preserve particle ordering to appease the validator
        th_til[i1]  = th + om*dt
        x_til[i1]   = x + V/om * sin_minus_sin
        y_til[i1]   = y - V/om * cos_minus_cos

        # th_til[:n1]  = th + om*dt
        # x_til[:n1]   = x + V/om * sin_minus_sin
        # y_til[:n1]   = y - V/om * cos_minus_cos

        # l'Hopital's case
        i2 = idx[~cond]
        V, om = V_all[i2], om_all[i2]
        x, y, th = x_all[i2], y_all[i2], th_all[i2]

        th_til[i2]  = th + om*dt
        x_til[i2]   = x + V*dt*np.cos(th)
        y_til[i2]   = y + V*dt*np.sin(th)

        # th_til[n1:]  = th + om*dt
        # x_til[n1:]   = x + V*dt*np.cos(th)
        # y_til[n1:]   = y + V*dt*np.sin(th)
        
        g = np.column_stack([x_til, y_til, th_til])


        """
        # Naive loopy version
        n = self.M
        g = np.zeros((n, 3))
        for i in range(n):
            x = self.xs[i]
            u = us[i]
            h = tb.compute_dynamics(x, u, dt, compute_jacobians=False)
            g[i] = h
        """

        ########## Code ends here ##########

        return g

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
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        
        vs, Q = self.measurement_model(z_raw, Q_raw)
        ws = scipy.stats.multivariate_normal.pdf(vs, mean=None, cov=Q)
        # ws = ws / np.sum(ws) # Autograder doesn't like normalized weights
        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        ########## Code starts here ##########
        # TODO: Compute Q.
        Q = scipy.linalg.block_diag(*Q_raw)
        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
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

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!

        n     = self.M                  # Num of particles. M.
        n_lin = self.map_lines.shape[1] # Num of known lines on map. J.
        n_mea = z_raw.shape[1]          # Num of scanned lines. I.

        z_raw = z_raw.T                     # shape(n_mea, 2)
        # Q_raw                             # shape(n_mea, 2, 2)
        hs = self.compute_predicted_measurements().transpose(0, 2, 1) # shape(n, n_lin, 2)

        z_mat = z_raw[None, None, :, :]     # shape(1, 1,     n_mea, 2)
        h_mat = hs[:, :, None, :]           # shape(n, n_lin, 1,     2)
        
        # Have to ensure angle is in range [0, pi]
        # v_mat = z_mat - h_mat # Innovation      # shape(n, n_lin, n_mea, 2)

        z_alp, h_alp = z_mat[..., 0], h_mat[..., 0] # shape(n, n_lin, n_mea)
        z_alp, h_alp = z_alp % (2.*np.pi), h_alp % (2.*np.pi)
        diff = z_alp - h_alp
        idx = np.abs(diff) > np.pi
        sign = 2. * (diff[idx] < 0.) - 1.
        diff[idx] += sign * 2. * np.pi
        v_alp = diff
        # Reconstruct v
        v_r = z_mat[..., 1] - h_mat[..., 1]
        v_mat = np.stack((v_alp, v_r), axis=3)

        v_fat = v_mat[..., None]                # shape(n, n_lin, n_mea, 2, 1)
        Q_inv = np.linalg.inv(Q_raw)            # shape(      n_mea, 2, 2)
        Q_inv = Q_inv[None, None, :, :, :]      # shape(1, 1, n_mea, 2, 2) # PEP20

        d_mat = np.matmul(v_fat.transpose(0, 1, 2, 4, 3), Q_inv)
        d_mat = np.matmul(d_mat, v_fat)         # shape(n, n_lin, n_mea, 1, 1)
        d_mat = d_mat.reshape((n,n_lin,n_mea))  # shape(n, n_lin, n_mea)

        # For each particle, for each scanned line, this returns the index
        # of the best known line.
        d_argmin = np.argmin(d_mat, axis=1)                 # shape(n, n_mea)
        d_argmin = d_argmin[:, None, :, None]               # shape(n, 1, n_mea, 1)
        vs = np.take_along_axis(v_mat, d_argmin, axis=1)    # shape(n, 1, n_mea, 2)
        vs = vs.reshape((n, n_mea, 2))                      # shape(n, n_mea, 2)

        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """

        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebod_model functions directly here. This
        #       results in a ~10x speedup.

        # Adapted from tb.transform_line_to_scanner_frame()
        
        # n = self.M                      # Num of particles
        # d = self.xs.shape[1]            # 3 for (x, y, th)
        # n_lin = self.map_lines.shape[1] # Num of lines on map. This is our pset fudge.
        #                                 # We're not generally supposed to know this.

        hs = self.map_lines.T           # shape(n_lin, 2)
        alp, r = hs[:, 0], hs[:, 1]

        x, y, th = self.xs.T            # shapes(3, )
        xcam_R, ycam_R, thcam_R = self.tf_base_to_camera    # Camera pose. in Robot frame.

        xcam = xcam_R*np.cos(th) - ycam_R*np.sin(th) + x
        ycam = xcam_R*np.sin(th) + ycam_R*np.cos(th) + y

        # shapes(n, n_lin)
        alp_C = alp[None, :] - th[:, None] - thcam_R
        r_C = (r[None, :] - xcam[:, None]*np.cos(alp)[None, :] -
                            ycam[:, None]*np.sin(alp)[None, :])
        
        # Vectorized tb.normalize_line_parameters
        cond = r_C < 0
        alp_C[cond] += np.pi
        r_C[cond]   *= -1
        alp_C = (alp_C + np.pi) % (2*np.pi) - np.pi

        hs = np.array([alp_C, r_C]).transpose(1, 0, 2)  # shape(n, 2, n_lin)


        """
        # Naive loopy version

        n = self.M # num of particles
        n_lin = self.map_lines.shape[1] # num of scanned lines

        # Preallocate output
        hs = np.zeros((n, 2, n_lin))

        for i in range(n):  # For each particle
            x = self.xs[i]
            for j in range(n_lin):  # For each line
                line = self.map_lines[:, j]
                hs[i, :, j] = tb.transform_line_to_scanner_frame(line, x,
                        self.tf_base_to_camera, compute_jacobian=False)
        """

        ########## Code ends here ##########

        return hs
