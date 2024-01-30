import numpy as np
from src.utils.utils import adjoint_transformation, matrix_logarithm
from src.fk.forward_kinematics import forward_kinematics
from src.jacobian.space_jacobian import space_jacobian

def inverse_kinematics(M, S, T_goal, q_initial, threshold=1e-3, max_iterations=100):
    """
    Compute the inverse kinematics for a robotic manipulator using a numerical approach (such as the Newton-Raphson method).

    Parameters:
    M : numpy.ndarray
        The home configuration (position and orientation) of the end-effector.
    S : List[numpy.ndarray]
        A list of screw axes in the space frame, one for each joint.
    T_goal : numpy.ndarray
        The desired end-effector configuration (position and orientation).
    q_initial : numpy.ndarray
        An initial guess of joint angles.
    threshold : float
        The threshold for stopping the iterations.
    max_iterations : int
        The maximum number of iterations to perform.

    Returns:
    numpy.ndarray
        The joint angles that achieve the desired end-effector configuration.
    """
    
    # Example screw axes, home configuration, and desired end-effector configuration
    # S = [...]
    # M = np.array([[...]])
    # T_goal = np.array([[...]])
    # q_initial = np.array([0, 0, 0, 0])  # Initial guess for joint angles
    # 
    # Calculate inverse kinematics
    # q_solution = inverse_kinematics(M, S, T_goal, q_initial)

    
    q = np.array(q_initial)
    for _ in range(max_iterations):
        T_current = forward_kinematics(M, S, q)

        # Calculate the error between current and goal transformations
        T_error = np.dot(np.linalg.inv(T_current), T_goal)
        Vb = matrix_logarithm(T_error)
        Vb_se3 = np.array([Vb[2, 1], Vb[0, 2], Vb[1, 0], Vb[0, 3], Vb[1, 3], Vb[2, 3]])
        
        # Check for convergence
        if np.linalg.norm(Vb_se3) < threshold:
            return q

        # Convert space Jacobian to body Jacobian
        J_space = space_jacobian(S, q)
        Ad_T_current = adjoint_transformation(T_current)
        J_body = np.dot(np.linalg.inv(Ad_T_current), J_space)

        # Update joint variables
        q += np.linalg.pinv(J_body) @ Vb_se3

    raise ValueError("Inverse kinematics did not converge")
