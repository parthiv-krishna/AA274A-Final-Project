import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    t = np.zeros(len(path)) # nominal times that we reach each point
    for i in range(1, len(path)): # skip t[0] which we know is 0
        curr_pt = np.array(path[i])
        prev_pt = np.array(path[i-1])
        dist_traveled = np.linalg.norm(curr_pt - prev_pt) # straight line distance
        delta_t = dist_traveled / V_des # distance / velocity = time
        t[i] = t[i-1] + delta_t
    
    x = np.array(path)[:,0]
    y = np.array(path)[:,1]
    t_smoothed = np.arange(t[0], t[-1], dt)

    # interpolate x and y as functions of (nominal) time
    tck_x = scipy.interpolate.splrep(t, x, s=alpha)
    tck_y = scipy.interpolate.splrep(t, y, s=alpha)

    # create traj_smoothed array
    traj_smoothed = np.zeros([len(t_smoothed),7])
    traj_smoothed[:,0] = scipy.interpolate.splev(t_smoothed, tck_x)         # x
    traj_smoothed[:,1] = scipy.interpolate.splev(t_smoothed, tck_y)         # y
    traj_smoothed[:,3] = scipy.interpolate.splev(t_smoothed, tck_x, der=1)  # xd 
    traj_smoothed[:,4] = scipy.interpolate.splev(t_smoothed, tck_y, der=1)  # yd
    traj_smoothed[:,2] = np.arctan2(traj_smoothed[:,4], traj_smoothed[:,3]) # th = arctan(yd/xd)
    traj_smoothed[:,5] = scipy.interpolate.splev(t_smoothed, tck_x, der=2)  # xdd
    traj_smoothed[:,6] = scipy.interpolate.splev(t_smoothed, tck_y, der=2)  # ydd
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
