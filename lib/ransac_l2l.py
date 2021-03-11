# fitting a ransac model to plucker line correspondences
import random
import numpy as np


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


def model_estimate(plucker1, plucker2):
    # estimate the rotation and translation using the plucker line correpsondences
    M = np.zeros((3, 3), dtype=np.float32)
    for i in range(2):
        M += np.matmul(plucker2[3:, i].reshape(3, 1), plucker1[3:, i].reshape(1, 3))

    u, s, vh = np.linalg.svd(M, full_matrices=True)

    rotation_est = np.matmul(u, vh)
    # fix the rotation det to 1
    rotation_est = rotation_est / np.linalg.det(rotation_est)

    # solve the translation vector
    A = np.zeros((6, 3), dtype=np.float32)
    b = np.zeros((6, 1), dtype=np.float32)

    A[:3, :] = np.transpose(skew(np.matmul(rotation_est, plucker1[3:, 0].reshape(3, 1))))
    A[3:, :] = np.transpose(skew(np.matmul(rotation_est, plucker1[3:, 1].reshape(3, 1))))

    b[:3, :] = plucker2[:3, 0].reshape(3, 1) - np.matmul(rotation_est, plucker1[:3, 0].reshape(3, 1))
    b[3:, :] = plucker2[:3, 1].reshape(3, 1) - np.matmul(rotation_est, plucker1[:3, 1].reshape(3, 1))

    trans_est = np.matmul(np.linalg.pinv(A), b)

    return rotation_est, trans_est


def score(plucker1, plucker2, rotation_est, trans_est, threshold):

    # estimated the number of inliers given the estimated rotation and translation
    _, N = plucker1.shape

    # Let's form the line motion matrix
    motion_matrix = np.zeros((6, 6), dtype=np.float32)
    motion_matrix[:3, :3] = rotation_est
    motion_matrix[:3, 3:] = np.matmul(skew(trans_est), rotation_est)
    motion_matrix[3:, 3:] = rotation_est

    # given the line motion matrix,let's move the plucker lines in frame 1 to frame 2
    plucker1_moved = np.matmul(motion_matrix, plucker1)

    # --------------------------------------------------------------------
    # using the l2 distance
    distance = np.linalg.norm(plucker2 - plucker1_moved, axis=0)

    # thres the distance

    inlier_mask = distance < threshold

    return inlier_mask

def run_ransac(plucker1, plucker2, max_iterations=200, inlier_threshold=5e-1, random_seed=None):
    best_ic = 0 # best inlier count
    best_ic_mask = None # best inlier mask
    best_rot, best_trans = None, None # return the best model estimated
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array

    dim, N = plucker1.shape

    for i in range(max_iterations):
        # minimal 2 matches need to be sampled
        selected_ind = np.random.choice(N, 2)

        selected_plucker1 = plucker1[:, selected_ind]
        selected_plucker2 = plucker2[:, selected_ind]

        rotation_est, trans_est = model_estimate(selected_plucker1, selected_plucker2)

        # given the estimated rotation and translation, let's estimated the number of inliers

        inlier_mask_cur = score(plucker1, plucker2, rotation_est, trans_est, inlier_threshold)
        nb_inliers_cur = np.sum(inlier_mask_cur)

        if nb_inliers_cur > best_ic:
            best_ic = nb_inliers_cur
            best_ic_mask = inlier_mask_cur
            best_rot, best_trans = rotation_est, trans_est

    # perform inlier set optimization
    # given the inlier correspondences, let's estimate the final rotation and translation
    if best_ic_mask is not None and best_ic > 1:
        plucker1_inlier = plucker1[:, best_ic_mask]
        plucker2_inlier = plucker2[:, best_ic_mask]
        best_rot, best_trans = best_fit_transform(plucker1_inlier, plucker2_inlier)

    return best_rot, best_trans, best_ic, best_ic_mask