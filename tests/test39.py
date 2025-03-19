import torch

# assume there is some other function that can write out all the symbols needed for this function
func_str = """
def axa(a, x):
    return torch.stack([
        a[0] * a[0] * x[0] + 2.0 * a[0] * a[1] * x[1] - a[1] * a[1] * x[0],
        a[1] * a[1] * x[1] + 2.0 * a[0] * a[1] * x[0] - a[0] * a[0] * x[1]
    ])
"""

exec(func_str)

batched_axa = torch.vmap(axa)

a_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x_batch = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

out_batch = batched_axa(a_batch, x_batch)
print(out_batch)
