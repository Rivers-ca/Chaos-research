import torch 


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    row_max = torch.max(logits, dim=1, keepdim = True).values
    shifted_logits = logits - row_max
    exp_logits = torch.exp(shifted_logits)
    sum_exp = exp_logits.sum(dim = 1, keepdim = True)
    log_prob = shifted_logits - torch.log(sum_exp)
    batch_indices = torch.arange (logits.shape[0], device = logits.device)
    return -log_prob[batch_indices, targets].mean()

if __name__ == "__main__" :
    test_normal_logits = torch.tensor(
        [[2.0, 1.0, 0.0],
         [0., 1., 3.]],
        dtype = torch.float32
    )
    test_normal_targets = torch.tensor(
        [[0, 2]],
        dtype = torch.int64
    )
    test_extreme_logits = torch.tensor(
            [[1000.0, 0.0]],
            dtype = torch.float32
        )
    test_extreme_targets = torch.tensor(
                [[0]],
                dtype = torch.int64
            )
    print("normal: ", cross_entropy(test_normal_logits, test_normal_targets))
    print("extreme: ", cross_entropy(test_extreme_logits, test_extreme_targets))