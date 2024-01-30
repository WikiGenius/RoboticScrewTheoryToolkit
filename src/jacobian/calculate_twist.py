import numpy as np

def calculate_twist(J, q_dot):
    """
    Calculate the twist of a robotic manipulator.

    Parameters:
    J : numpy.ndarray
        The Jacobian matrix of the manipulator.
    q_dot : numpy.ndarray
        The joint velocities vector.

    Returns:
    numpy.ndarray
        The twist vector representing the end-effector's velocity.
    """
    
    # Example Jacobian matrix
    # J = np.array([[1, 0, 0],
                #   [0, 1, 0],
                #   [0, 0, 1],
                #   [0, 0, 0],
                #   [0, 0, 0],
                #   [1, 0, 0]])
    # 
    # Example joint velocities
    # q_dot = np.array([1, 0.5, 0.2])
    # 
    # Calculate the twist
    # twist = calculate_twist(J, q_dot)

    
    if J.shape[1] != len(q_dot):
        raise ValueError("Dimension mismatch between Jacobian and joint velocities")

    return np.dot(J, q_dot)
