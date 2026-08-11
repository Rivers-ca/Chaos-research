import torch 

def masked_mean(x: torch.Tensor, mask: torch.Tensor):

    masked_x = x * mask
    row_sum = masked_x.sum(dim = 1)
    mask_sum = mask.sum(dim = 1)
    safe_mask = mask_sum.clamp(min = 1)
    return row_sum / safe_mask

if __name__ == "__main__":

    test = torch.tensor([
        [1., 2., 99.,],
        [5., 5., 5.,],
                        ],
        dtype = torch.float32
)
    mask = torch.tensor ([
        [True, True, False],
        [False, False, False]
                         ])

    print(masked_mean(test, mask)
          )