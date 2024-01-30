import numpy as np
from numpy.linalg import pinv

def calculate_joint_velocities(J, V_s):
    """
    Calculate the joint velocities for a given end-effector twist and Jacobian.

    Parameters:
    J : numpy.ndarray
        The Jacobian matrix J_s(q).
    V_s : numpy.ndarray
        The end-effector twist \mathcal{V}_s.

    Returns:
    numpy.ndarray
        The joint velocities \dot{q}.
    """
    # Example Jacobian matrix
    # J = np.array([[1, 0, 0],
                #   [0, 1, 0],
                #   [0, 0, 1],
                #   [0, 0, 0],
                #   [0, 0, 0],
                #   [1, 0, 0]])
    # 
    # Desired end-effector twist
    # V_s = np.array([0.5, 0.1, 0.2, 0, 0, 1])
    # 
    # Calculate joint velocities
    # q_dot = calculate_joint_velocities(J, V_s)


    # Ensure J is a square matrix or has more rows than columns
    if J.shape[0] < J.shape[1]:
        raise ValueError("The Jacobian matrix must have more rows than columns or be square.")

    # Compute the pseudo-inverse of the Jacobian
    J_pseudo_inverse = pinv(J)

    # Calculate the joint velocities
    q_dot = np.dot(J_pseudo_inverse, V_s)

    return q_dot
