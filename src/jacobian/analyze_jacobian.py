import numpy as np

def analyze_jacobian(J):
    """
    Analyze the Jacobian matrix of a robotic manipulator.

    Parameters:
    J : numpy.ndarray
        The Jacobian matrix.

    Returns:
    dict
        A dictionary containing the determinant (if applicable), rank, and singularity status.
    """
    
    # Example Jacobian matrix
    # J = np.array([[1, 0, 0],
    #               [0, 0, 1],
    #               [0, -1, 0]])

    # Analyze the Jacobian
    # analysis_result = analyze_jacobian(J)


    analysis = {}
    m, n = J.shape

    # Determinant (only for square matrices)
    if m == n:
        analysis['determinant'] = np.linalg.det(J)
    else:
        analysis['determinant'] = None

    # Rank of the Jacobian
    analysis['rank'] = np.linalg.matrix_rank(J)

    # Singularity check
    if m == n and analysis['determinant'] == 0:
        analysis['is_singular'] = True
    elif m != n and analysis['rank'] < min(m, n):
        analysis['is_singular'] = True
    else:
        analysis['is_singular'] = False

    return analysis
