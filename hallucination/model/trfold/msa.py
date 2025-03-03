import torch
import torch.nn.functional as F


def msa_corruption(msa):
    D, L = msa.shape
    mask_pos = torch.empty(D, L).uniform_(0, 1)
    i, j = torch.where((mask_pos >= 0.12) & (mask_pos < 0.135))
    i_random = torch.randint(low=0, high=D, size=i.shape)
    msa[i, j] = msa[i_random, j]
    i, j = torch.where((mask_pos >= 0.105) & (mask_pos < 0.12))
    msa[i, j] = torch.randint(low=0, high=21, size=i.shape)
    msa[mask_pos < 0.105] = 21
    mask_pos = torch.where(mask_pos < 0.15, 1, 0)
    return msa, mask_pos


def msa_cluster(cluster, extra):
    cluster_num, L = cluster.shape
    extra_num, L = extra.shape
    weights = torch.cat([torch.ones(20), torch.zeros(2)])
    cluster_one_hot = F.one_hot(cluster, num_classes=22).float()
    extra_one_hot = F.one_hot(extra, num_classes=22).float()
    agreement = torch.matmul(
        torch.reshape(extra_one_hot, [extra_num, L * 22]),
        torch.reshape(cluster_one_hot * weights, [cluster_num, L * 22]).transpose(0, 1),
    )
    agreement = torch.argmax(agreement, dim=1)

    def csum(x, alignment, cluster_num):
        extra_num, L, C = x.shape
        num_aligns = torch.max(alignment)
        dims = len(x.shape)
        align_shape = [1 for i in range(dims)]
        align_shape[0] = -1
        return torch.zeros(cluster_num, L, C).scatter_add_(
            dim=0,
            index=alignment.reshape(align_shape).expand([-1] + list(x.shape)[1:]),
            src=x,
        )

    profile = csum(extra_one_hot, agreement, cluster_num)
    profile = profile / (profile.sum(dim=-1, keepdim=True) + 1e-6)
    return profile


def msa_parser(msa, cluster_num, extra_num, bf16=True):
    D, L = msa.shape
    seq = msa[:1, :].clone()
    if D > cluster_num:
        idx = torch.randperm(D - 1) + 1
        msa = torch.cat([seq, msa[idx]])
        cluster = msa[:cluster_num, :].clone()
        extra = msa[cluster_num:, :].clone()
    else:
        cluster = msa.clone()
        extra = msa[:1, :].clone()
    msa_gt = cluster.clone()
    cluster, mask_pos = msa_corruption(cluster)
    profile = msa_cluster(cluster, extra)
    cluster = F.one_hot(cluster, num_classes=22).float()
    cluster = torch.cat([cluster, profile], dim=-1)
    msa_gt[mask_pos == 0] = -100
    if extra.shape[0] > extra_num:
        extra = extra[:extra_num]
    if bf16:
        cluster = cluster.bfloat16()

    return cluster, extra, msa_gt
