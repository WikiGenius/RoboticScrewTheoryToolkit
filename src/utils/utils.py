import numpy as np
from scipy.linalg import expm, logm

def skew_symmetric(omega):
    """
    Create a skew-symmetric matrix from a 3-element vector omega.

    Parameters:
    omega : array_like
        A 3-element vector.

    Returns:
    numpy.ndarray
        A 3x3 skew-symmetric matrix.
    """
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])

def screw_axis(omega_v, a, jt):
    """
    Calculate the screw axis.

    Parameters:
    omega_v : array_like
        The angular velocity component (omega) for revolute joints or 
        linear velocity component (v) for prismatic joints.
    a : coordinate vector [x, y, z]
        The position vector of a point on the joint axis
    jt : str
        Joint type ('R' for revolute, 'P' for prismatic).

    Returns:
    numpy.ndarray
        The screw axis.
    """
    # Example usage
    # omega = [1, 0, 0] for a revolute joint
    # v = [0, 0, 1] for a prismatic joint
    # a = [0, 0, 0] as a position vector
    # print(screw_axis(omega, a, 'R'))  # Example call for a revolute joint
    # print(screw_axis(v, a, 'P'))      # Example call for a prismatic joint
    
    if jt == 'R':
        # For revolute joints, omega_v represents the angular velocity omega
        return np.hstack((omega_v, np.cross(a, omega_v)))
    elif jt == 'P':
        # For prismatic joints, omega_v represents the linear velocity v
        return np.hstack((np.zeros(3), omega_v))
    else:
        raise ValueError("Joint type must be 'R' or 'P'.")


def adjoint_transformation(T):
    """
    Compute the adjoint transformation matrix for a given transformation matrix T.

    Parameters:
    T : array_like
        A 4x4 transformation matrix.

    Returns:
    numpy.ndarray
        A 6x6 adjoint transformation matrix.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    p_skew = skew_symmetric(p)
    adj_T = np.zeros((6, 6))
    adj_T[0:3, 0:3] = R
    adj_T[3:6, 3:6] = R
    adj_T[3:6, 0:3] = np.dot(p_skew, R)
    return adj_T

def exponential_map(S, theta):
    """
    Compute the exponential map of a screw axis.

    Parameters:
    S : array_like
        The screw axis.
    theta : float
        The joint variable.

    Returns:
    numpy.ndarray
        The exponential map (transformation matrix).
    """
    return expm(np.array([[0, -S[2], S[1], S[3]],
                          [S[2], 0, -S[0], S[4]],
                          [-S[1], S[0], 0, S[5]],
                          [0, 0, 0, 0]]) * theta)

def matrix_logarithm(T):
    """
    Compute the matrix logarithm of a transformation matrix.

    Parameters:
    T : array_like
        A 4x4 transformation matrix.

    Returns:
    numpy.ndarray
        The matrix logarithm of T.
    """
    return logm(T)

def rotation_matrix(axis, theta):
    """
    Create a rotation matrix corresponding to the rotation around a general axis by a specified angle.

    Parameters:
    axis : array_like
        A 3-element array representing the rotation axis.
    theta : float
        The rotation angle in radians.

    Returns:
    numpy.ndarray
        A 3x3 rotation matrix.
    """
    # Normalize the axis vector
    axis = np.asarray(axis)
    axis = normalize_vector(axis)

    # Create a skew-symmetric matrix from axis
    axis_skew = skew_symmetric(axis)

    # Use scipy.linalg.expm to compute the exponential map
    R = expm(axis_skew * theta)
    return R

    
def normalize_vector(v):
    """
    Normalize a vector.

    Parameters:
    v : array_like
        A vector.

    Returns:
    numpy.ndarray
        A normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


# Add additional utility functions as needed
