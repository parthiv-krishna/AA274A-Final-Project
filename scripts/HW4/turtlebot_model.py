import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(x, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                        x: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    ########## Code starts here ##########

    # For derivations see answer to written HW4 Q1(i)

    X = x # Rename
    V, om = u       # u_t
    x, y, th = X    # x_{t-1}
    
    thomdt = th+om*dt # Common term

    if np.absolute(om) > EPSILON_OMEGA:
        # Common terms
        sin_minus_sin = np.sin(thomdt) - np.sin(th)
        cos_minus_cos = np.cos(thomdt) - np.cos(th)

        th_til  = thomdt
        x_til   = x + V/om * sin_minus_sin
        y_til   = y - V/om * cos_minus_cos

        dx_dth  = V/om * cos_minus_cos
        dy_dth  = V/om * sin_minus_sin
        dx_dV   = sin_minus_sin / om
        dy_dV   = -cos_minus_cos / om
        dx_dom  = V/(om*om) * (-sin_minus_sin + om*dt*np.cos(thomdt))
        dy_dom  = V/(om*om) * ( cos_minus_cos + om*dt*np.sin(thomdt))
        
    else:
        th_til  = thomdt
        x_til   = x + V*dt*np.cos(th)
        y_til   = y + V*dt*np.sin(th)

        dx_dth  = -V*dt*np.sin(th)
        dy_dth  =  V*dt*np.cos(th)
        dx_dV   = dt*np.cos(th)
        dy_dV   = dt*np.sin(th)
        dx_dom  = -0.5*V*dt*dt*np.sin(th)
        dy_dom  =  0.5*V*dt*dt*np.cos(th)

    g = np.array([x_til, y_til, th_til]) # x_t = g is approximated by x_tilde

    Gx = np.array([
        [1, 0, dx_dth],
        [0, 1, dy_dth],
        [0, 0, 1]
    ], dtype=np.float)

    Gu = np.array([
        [dx_dV, dx_dom],
        [dy_dV, dy_dom],
        [0,     dt]
    ], dtype=np.float)

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx

    # For derivation see written part (iii)

    alpha, r = normalize_line_parameters(line)

    alp = alpha # Rename
    X = x

    # So, we need to express the given line from World frame to Camera frame.
    # Right now our camera pose is expressed wrt Robot frame, so first let's
    # get that in terms of World coordinates.

    # Frames other than world frame carry a suffix.
    x, y, th = X                                  # Mean of belief of robot pose wrt World frame.
    xcam_R, ycam_R, thcam_R = tf_base_to_camera   # Camera pose. in Robot frame.

    """
    C_Rh = np.array([xcam_R, ycam_R, 1])          # Camera pose in homogeneous coords. Robot frame.

    # Homogeneous transform matrix from Robot frame coords to World frame coords
    # This is the matrix {}^R_W T
    RW_T = np.array([
        [np.cos(th), -np.sin(th), x],
        [np.sin(th),  np.cos(th), y],
        [0, 0, 1]
    ])

    Cam_h = np.matmul(RW_T, C_Rh) # Camera in homogeneous coords. World frame.
    xcam, ycam, hcam = Cam_h
    xcam, ycam = xcam/hcam, ycam/hcam  # Camera in World frame coords.
    """

    xcam = xcam_R*np.cos(th) - ycam_R*np.sin(th) + x
    ycam = xcam_R*np.sin(th) + ycam_R*np.cos(th) + y

    # From the diagram, alp_C (alpha in Camera coords)
    # is just alp_W subtracting (compensating for) relative frame rotations
    # between W -> R -> C
    alp_C = alp - th - thcam_R
    # Since we have normalized the line parameters r_C is always positive
    # so the absolute value disappears.
    r_C = r - xcam*np.cos(alp) - ycam*np.sin(alp)

    h = np.array([alp_C, r_C])

    # Hx is asking how (alp_C, r_C) changes when the base changes position.
    dalpC_dxR  = 0 # alpha doesn't care about x, y
    dalpC_dyR  = 0
    dalpC_dthR = -1

    drC_dxW = -np.cos(alp)
    drC_dyW = -np.sin(alp)
    drC_dthW = (-np.cos(alp) * (-xcam_R*np.sin(th) - ycam_R*np.cos(th))
                -np.sin(alp) * ( xcam_R*np.cos(th) - ycam_R*np.sin(th))
                )

    Hx = np.array([
        [dalpC_dxR, dalpC_dyR, dalpC_dthR],
        [drC_dxW,   drC_dyW,   drC_dthW]
    ])

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
