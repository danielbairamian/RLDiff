import torch
import math

@torch.no_grad()
def sample_iadb_linear_first_order(iadb_model, x0, nb_step, return_trajectory=False):
    T = nb_step
    B = x0.shape[0]
    
    x0_states = torch.zeros((T+1, B, *x0.shape[1:]))  # CPU
    alphas = torch.ones(T+1, B)
    x0_states[0] = x0
    alphas[0] = torch.tensor(0.0).repeat(B)

    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = iadb_model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d
        x0_states[t+1] = x_alpha.cpu()  # Move to CPU for storage
        alphas[t+1] = torch.tensor(alpha_end).repeat(B)  # Store alpha for each sample in the batch

    if return_trajectory:
        return x_alpha, {"states": x0_states, "alphas": alphas} 
    return x_alpha

@torch.no_grad()
def sample_iadb_cosine_first_order(iadb_model, x0, nb_step, return_trajectory=False):

    T = nb_step
    B = x0.shape[0]
    
    x0_states = torch.zeros((T+1, B, *x0.shape[1:]))  # CPU
    alphas = torch.ones(T+1, B)
    x0_states[0] = x0
    alphas[0] = torch.tensor(0.0).repeat(B)

    x_alpha = x0
    for t in range(nb_step):
        alpha_start = 1 - math.cos((t/nb_step) * math.pi / 2)
        alpha_end = 1 - math.cos(((t+1)/nb_step) * math.pi / 2)

        d = iadb_model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

        x0_states[t+1] = x_alpha.cpu()  # Move to CPU for storage
        alphas[t+1] = torch.tensor(alpha_end).repeat(B)  # Store alpha for each sample in the batch
    
    if return_trajectory:
        return x_alpha, {"states": x0_states, "alphas": alphas}
    return x_alpha

@torch.no_grad()
def sample_iadb_linear_second_order(iadb_model, x0, nb_step, return_trajectory=False):
    nb_step = nb_step // 2

    T = nb_step
    B = x0.shape[0]
    
    x0_states = torch.zeros((T+1, B, *x0.shape[1:]))  # CPU
    alphas = torch.ones(T+1, B)
    x0_states[0] = x0
    alphas[0] = torch.tensor(0.0).repeat(B)

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

        x0_states[t+1] = x_alpha.cpu()  # Move to CPU for storage
        alphas[t+1] = torch.tensor(alpha_end).repeat(B)  # Store alpha for each sample in the batch

    if return_trajectory:
        return x_alpha, {"states": x0_states, "alphas": alphas}
    return x_alpha

@torch.no_grad()
def sample_iadb_cosine_second_order(iadb_model, x0, nb_step, return_trajectory=False):
    nb_step = nb_step // 2

    T = nb_step
    B = x0.shape[0]
    
    x0_states = torch.zeros((T+1, B, *x0.shape[1:]))  # CPU
    alphas = torch.ones(T+1, B)
    x0_states[0] = x0
    alphas[0] = torch.tensor(0.0).repeat(B)

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

        x0_states[t+1] = x_alpha.cpu()  # Move to CPU for storage
        alphas[t+1] = torch.tensor(alpha_end).repeat(B)  # Store alpha for each sample in the batch 
    
    if return_trajectory:
        return x_alpha, {"states": x0_states, "alphas": alphas}
    return x_alpha