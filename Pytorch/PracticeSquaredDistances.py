import torch

def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor):
    a_exp = a.unsqueeze(1)
    b_exp = b.unsqueeze(0)
    dif = a_exp - b_exp
    sq = dif*dif 
    sq_dist = sq.sum(dim = 2)
    return sq_dist


if __name__ == "__main__":
    ac = torch.tensor([[1., 2.,],
                        [3., 4.,]],
                         dtype = torch.float32)
    bc = torch.tensor([[1., 2.,],
                         [3., 4.,]],
                         dtype = torch.float32)
    a, b = torch.randn(50, 8), torch.randn(30, 8)

    print(pairwise_sq_dist(a, b))