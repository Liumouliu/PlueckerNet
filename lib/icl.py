# Iterative Closest Line method.


import numpy as np
from sklearn.neighbors import NearestNeighbors

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def best_fit_transform(plucker1, plucker2):
    '''
    Calculates the least-squares best-fit transform that maps corresponding Lines plucker1 to plucker2  in m spatial dimensions
    '''

    assert plucker1.shape == plucker2.shape

    nb_lines = plucker1.shape[1]

    M = np.zeros((3, 3), dtype=np.float32)

    for i in range(nb_lines):
        M += np.matmul(plucker2[3:, i].reshape(3, 1), plucker1[3:, i].reshape(1, 3))

    u, s, vh = np.linalg.svd(M, full_matrices=True)

    rotation_est = np.matmul(u, vh)
    # fix the rotation det to 1
    rotation_est = rotation_est / np.linalg.det(rotation_est)

    # solve the translation vector
    A = np.zeros((3*nb_lines, 3), dtype=np.float32)
    b = np.zeros((3*nb_lines, 1), dtype=np.float32)

    for i in range(nb_lines):
        start_ind = 3*i
        end_ind = 3*(i+1)
        A[start_ind:end_ind, :] = np.transpose(skew(np.matmul(rotation_est, plucker1[3:, i].reshape(3, 1))))
        b[start_ind:end_ind, :] = plucker2[:3, i].reshape(3, 1) - np.matmul(rotation_est, plucker1[:3, i].reshape(3, 1))

    trans_est = np.matmul(np.linalg.pinv(A), b)

    return rotation_est, trans_est



# line distance using https://hal.archives-ouvertes.fr/hal-00092721/document

# def nearest_neighbor(src, dst):
#     '''
#     Find the nearest (Euclidean) neighbor in dst for each line in src
#     Input:
#         src: 6xm array of lines
#         dst: 6xn array of lines
#     Output:
#         distances: distances of the nearest neighbor
#         indices: dst indices of the nearest neighbor
#     '''
#     m = src.shape[1]
#     n = dst.shape[1]
#
#     distances = np.zeros((m, n), dtype=np.float32)
#     for i in range(m):
#         for j in range(n):
#             dis_mat = np.matmul(dst[:, j].reshape(6, 1), src[:, i].reshape(1, 6)) - np.matmul(src[:, i].reshape(6, 1), dst[:, j].reshape(1, 6))
#             # get the F-norm of the mat
#             distances[i, j] = np.linalg.norm(dis_mat, 'fro')
#
#     knn_ind = np.argmin(distances, axis=1)
#     knn_dis = np.take_along_axis(distances, np.expand_dims(knn_ind, axis=-1), axis=1)
#
#     return knn_dis.ravel(), knn_ind.ravel()


# Euclidean distance

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each line in src
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst.T)
    distances, indices = neigh.kneighbors(src.T, return_distance=True)
    return distances.ravel(), indices.ravel()



# given the rotation and translation, move the plucker line
def moved_lines(plucker, rotation_est, trans_est):
    # Let's form the line motion matrix
    motion_matrix = np.zeros((6, 6), dtype=np.float32)
    motion_matrix[:3, :3] = rotation_est
    motion_matrix[:3, 3:] = np.matmul(skew(trans_est), rotation_est)
    motion_matrix[3:, 3:] = rotation_est

    # given the line motion matrix,let's move the plucker lines in frame 1 to frame 2
    plucker1_moved = np.matmul(motion_matrix, plucker)

    return plucker1_moved

def icl(plucker1, plucker2, init_rot=None, init_trans=None, max_iterations=20, tolerance=0.001):


    assert plucker1.shape[0] == plucker2.shape[0]

    # get number of dimensions
    m = plucker1.shape[1]

    # copy them to maintain the originals
    plucker1_cp = np.copy(plucker1)
    plucker2_cp = np.copy(plucker2)

    # apply the initial pose estimation
    if init_rot is not None and init_trans is not None:
        plucker1_cp = moved_lines(plucker1_cp, init_rot, init_trans)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination lines
        distances, indices = nearest_neighbor(plucker1_cp, plucker2_cp)

        # compute the transformation between the current source and nearest destination points
        rotation_est, trans_est = best_fit_transform(plucker1_cp, plucker2_cp[:,indices])

        # update the current source
        plucker1_cp = moved_lines(plucker1_cp, rotation_est, trans_est)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation, we know the exact correspondences here!
    rotation_est, trans_est = best_fit_transform(plucker1, plucker1_cp)

    return rotation_est, trans_est, distances, i



def icl_trimmed(plucker1, plucker2, init_rot=None, init_trans=None, max_iterations=100, tolerance=0.001, min_trim_nb = 200):


    assert plucker1.shape[0] == plucker2.shape[0]

    # get number of dimensions
    m = plucker1.shape[1]

    # copy them to maintain the originals
    plucker1_cp = np.copy(plucker1)
    plucker2_cp = np.copy(plucker2)

    # apply the initial pose estimation
    if init_rot is not None and init_trans is not None:
        plucker1_cp = moved_lines(plucker1_cp, init_rot, init_trans)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination lines
        distances, indices = nearest_neighbor(plucker1_cp, plucker2_cp)
        best_matches = np.argsort(distances)

        min_index = min(best_matches.shape[0], min_trim_nb)

        trimed_ind = best_matches[:min_index]


        # compute the transformation between the current source and nearest destination points
        rotation_est, trans_est = best_fit_transform(plucker1_cp[:,trimed_ind], plucker2_cp[:,indices[trimed_ind]])

        # update the current source
        plucker1_cp = moved_lines(plucker1_cp, rotation_est, trans_est)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation, we know the exact correspondences here!
    rotation_est, trans_est = best_fit_transform(plucker1, plucker1_cp)

    return rotation_est, trans_est, distances, i

