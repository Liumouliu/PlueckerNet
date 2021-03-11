
import torch
import torch.nn.functional as F

def correspondenceProbabilityDistances(P, C):
    """ Difference between the probability mass assigned to inlier and
        outlier correspondences
    """
    return ((1.0 - 2.0 * C) * P).sum(dim=(-2, -1))

def correspondenceProbabilityBCE(P, C):
    """ BCE loss
    """
    num_pos = F.relu(C.sum(dim=(-2,-1))-1.0) + 1.0
    num_neg = F.relu((1.0- C).sum(dim=(-2,-1)) -1.0) + 1.0

    loss = ((P + 1e-20).log() * C).sum(dim=(-2,-1)) * 0.5 / num_pos
    loss += ((1.0 - P + 1e-20 ).log() * (1.0 - C)).sum(dim=(-2,-1)) * 0.5 / num_neg

    return -loss


def correspondenceLoss(P, C_gt):
    # Using precomputed C_gt
    return correspondenceProbabilityBCE(P, C_gt).mean() # [-1, 1)
#     return correspondenceProbabilityDistances(P, C_gt).mean() # [-1, 1)

class TotalLoss(torch.nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
    def forward(self, P, C_gt):
        loss = correspondenceLoss(P, C_gt).view(1)
        return loss



class RegressionLoss(torch.nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.weights_translation = 1.0
    def forward(self, pose_est, R_gt, t_gt):
        # convert the R_gt to quat
        quat_gt = rotation_matrix_to_quaternion(R_gt)
        # fix the sign of q0
        sel = (quat_gt[:, 0] < 0.0).float().view(-1, 1)
        quat_gt = (1.0 - sel) * quat_gt - sel * quat_gt

        R_loss = (quat_gt - pose_est[:,:4]).norm(dim=-1).mean()
        t_loss = (t_gt.squeeze(-1) - pose_est[:,4:]).norm(dim=-1).mean()

        loss = R_loss + self.weights_translation * t_loss

        return loss, R_loss, t_loss



def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
    """

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q





