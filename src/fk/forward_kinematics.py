import numpy as np
from scipy.linalg import expm
from src.utils.utils import exponential_map
def forward_kinematics(M, S, q):
    """
    Compute Forward Kinematics for a robotic manipulator using the Product of Exponentials formula.

    Parameters:
    M : numpy.ndarray
        The home configuration (position and orientation) of the end-effector.
    S : List[numpy.ndarray]
        A list of screw axes in the space frame, one for each joint.
    q : List[float]
        A list of joint variables (angles for revolute joints, displacements for prismatic joints).

    Returns:
    numpy.ndarray
        The transformation matrix representing the position and orientation of the end-effector.
    """
    # Home configuration (example)
    # M = np.array([[1, 0, 0, 0.5],
                #   [0, 1, 0, 0],
                #   [0, 0, 1, 0.1],
                #   [0, 0, 0, 1]])
    # 
    # Screw axes for each joint (example)
    # S1 = [0, 0, 1, 0, 0, 0]
    # S2 = [0, 1, 0, -0.1, 0, 0]
    # S = [S1, S2]
    # 
    # Joint variables (example)
    # q = [np.pi/2, 0.1]
    # 
    # Calculate FK
    # T_end_effector = forward_kinematics(M, S, q)

    
    T = np.eye(4)
    for i in range(len(S)):
        T = np.dot(T, exponential_map(S[i], q[i]))
    return np.dot(T, M)
