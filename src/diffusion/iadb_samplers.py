import torch
import math

@torch.no_grad()
def sample_iadb_linear_first_order(iadb_model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = iadb_model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

@torch.no_grad()
def sample_iadb_cosine_first_order(iadb_model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = 1 - math.cos((t/nb_step) * math.pi / 2)
        alpha_end = 1 - math.cos(((t+1)/nb_step) * math.pi / 2)

        d = iadb_model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

@torch.no_grad()
def sample_iadb_linear_second_order(iadb_model, x0, nb_step):
    nb_step = nb_step // 2
    x_alpha = x0

    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)
        alpha_mid = ((t+0.5)/nb_step)

        # intermediate step (x_t+1/2)
        d_1 = iadb_model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_mid = x_alpha + (alpha_mid-alpha_start)*d_1
        # second step (x_t+1)
        d_2 = iadb_model(x_mid, torch.tensor(alpha_mid, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d_2

    return x_alpha

@torch.no_grad()
def sample_iadb_cosine_second_order(iadb_model, x0, nb_step):
    nb_step = nb_step // 2
    x_alpha = x0

    for t in range(nb_step):
        alpha_start = 1 - math.cos((t/nb_step) * math.pi / 2)
        alpha_end = 1 - math.cos(((t+1)/nb_step) * math.pi / 2)
        alpha_mid = 1 - math.cos(((t+0.5)/nb_step) * math.pi / 2)

        # intermediate step (x_t+1/2)
        d_1 = iadb_model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_mid = x_alpha + (alpha_mid-alpha_start)*d_1
        # second step (x_t+1)
        d_2 = iadb_model(x_mid, torch.tensor(alpha_mid, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d_2
    
    return x_alpha