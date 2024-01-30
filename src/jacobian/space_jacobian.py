import numpy as np
from utils.utils import adjoint_transformation, exponential_map

def space_jacobian(S, q):
    """
    Compute the space Jacobian for a robotic manipulator.

    Parameters:
    S : List[numpy.ndarray]
        A list of screw axes in the space frame, one for each joint.
    q : List[float]
        A list of joint variables (angles for revolute joints, displacements for prismatic joints).

    Returns:
    numpy.ndarray
        The space Jacobian matrix.
    """
    # Screw axes for each joint (example)
    # S1 = np.array([0, 0, 1, 0, 0, 0])
    # S2 = np.array([0, 1, 0, -0.1, 0, 0])
    # S = [S1, S2]

    # Joint variables (example)
    # q = [np.pi/2, 0.1]

    # Calculate the space Jacobian
    # J_space = space_jacobian(S, q)

    
    num_joints = len(S)
    J = np.zeros((6, num_joints))
    T = np.eye(4)

    for i in range(num_joints):
        J[:, i] = np.dot(adjoint_transformation(T), S[i])
        T = np.dot(T, exponential_map(S[i], q[i]))

    return J
