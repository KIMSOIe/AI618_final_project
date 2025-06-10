import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import torch
import torch.nn.functional as F
import math

def poisson_likelihood_grad(x, y, H):
    Hx = H.H(x)
    grad = 1.0 - y / (Hx + 1e-6)
    return H.adjoint(grad)

def gamma_likelihood_grad(x, y, H, gamma=2.2):
    Hx = H.H(x).clamp(min=1e-2)  # 더 큰 최소값으로 수치 안정성 확보
    residual = y - Hx.pow(gamma)
    grad = gamma * residual / (Hx + 1e-6)
    grad = grad / grad.norm(p=2) * min(grad.norm(p=2), 50)
    print(grad.max())
    return H.adjoint(grad)

def clip_likelihood_grad(x, y, H, a=0.0, b=1.0):
    Hx = H.H(x)
    mask = ((Hx > a) & (Hx < b)).float()
    grad = (Hx - y) * mask
    return H.adjoint(grad)
def sr_likelihood_grad(x, y, H, sigma=0.05):
    Hx = H.H(x)
    residual = Hx - y
    grad = H.Ht(residual) / (sigma ** 2)
    return grad

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def langevin_posterior_sampling(
    x_init,
    model,
    beta,
    seq,
    y_0,
    H_func,
    num_steps=10,
    eps=1e-4,
    sigma_0=0.05,
    likelihood='poisson'
):
    x = x_init.clone().to(y_0.device)
    B = x.size(0)

    for t_idx in reversed(range(len(seq))):
        t = torch.full((B,), seq[t_idx], dtype=torch.long, device=x.device)
        alpha_t = compute_alpha(beta, t)
        for _ in tqdm(range(num_steps)):
            # Compute score from diffusion prior
            epsilon_pred = model(x, t)
            score_prior = -epsilon_pred / (1 - alpha_t).sqrt()

            # Compute data likelihood gradient
            if likelihood == 'poisson':
                grad_log_likelihood = poisson_likelihood_grad(x, y_0, H_func)
            elif likelihood == 'gamma':
                grad_log_likelihood = gamma_likelihood_grad(x, y_0, H_func)
            elif likelihood == 'clip':
                grad_log_likelihood = clip_likelihood_grad(x, y_0, H_func)
            elif likelihood == 'sr':
                grad_log_likelihood = sr_likelihood_grad(x, y_0, H_func, sigma=sigma_0)
            else:
                raise NotImplementedError(f"Unknown likelihood: {likelihood}")
            
            # Combine gradients
            grad = score_prior + grad_log_likelihood.view_as(score_prior)
            noise = torch.randn_like(x)

            # Langevin update
            x = x + 0.5 * eps * grad + (eps ** 0.5) * noise

    return x

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None,langevin_lr=1e-3, langevin_steps=1, langevin = False, langevin_noise = False, lambda_prior=0.07):
    with torch.no_grad():
        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas
        
        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes) ##score function
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  ## x_0_t 예측 , projection

            #variational inference conditioned on y 
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape) 

            '''command example: 
            python main.py --ni --config bedroom.yml --doc bedroom --timesteps 20 --eta 0.85 \
            --etaB 1 --deg cs2 --sigma_0 0.05 -i bedroom_cs2_noise0_sigma_0.05_langevin_step10_lr_1e-4 \
            --langevin --langevin_steps 10 --langevin_lr 1e-4'''

            # ------ Posterior Langevin Refinement ------
            if langevin: # Check if Langevin refinement is enabled

                # Temporarily enable gradient tracking for refinement
                with torch.enable_grad():
                    x_ref = xt_next.clone().detach().requires_grad_(True)

                    alpha_next = compute_alpha(b, next_t.long())  # ᾱ_{next_t}
                    sigma_t = (1 - alpha_next).sqrt().view(-1, 1, 1, 1)
                    sigma_t = sigma_t.clamp(min=1e-4) 

                    # Perform Langevin refinement for the specified number of steps
                    for _ in tqdm(range(langevin_steps)):
                        # Compute ∇_x log p(y | x) (likelihood term)
                        Hx = H_funcs.H(x_ref)
                        ll = - 0.5 * ((Hx - y_0).pow(2).flatten(1).sum(1) / (sigma_0**2)).sum()
                        grad_ll = torch.autograd.grad(ll, x_ref)[0]

                        # Optionally add Langevin noise
                        if langevin_noise:
                            # Compute ∇_x log p(x) (prior term) via the diffusion model
                            epsilon_pred = model(x_ref, next_t)  # ε̂θ(x_ref, next_t)
                            grad_prior = -epsilon_pred / (sigma_t)
                            # noise ~ N(0, I) scaled by sqrt(η)
                            noise = torch.randn_like(x_ref) * torch.sqrt(torch.tensor(langevin_lr, device=x.device))
                        else:
                            grad_prior = 0
                            noise = 0

                        # Combine the two gradients
                        total_grad = grad_ll + grad_prior * lambda_prior
                        # Langevin update: x ← x + η/2 (∇ log p(y|x)) + sqrt(η) * noise
                        x_ref = x_ref + (langevin_lr/2) * total_grad + noise
                        x_ref = x_ref.detach().requires_grad_(True)
                    xt_next = x_ref.detach()

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds

def stochastic_ddrm_sampling_aligned(    
    x,
    seq,
    model,
    b,
    H_funcs,
    y_0,
    sigma_0,
    etaB=1.0,
    etaA=1.0,
    etaC=1.0,
    langevin_scale=0.05,
    langevin_steps=1,
    cls_fn=None,
    classes=None,
):

    with torch.no_grad():
        B, C, H, W = x.shape
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(C * H * W, device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        # Initial projection
        largest_alphas = compute_alpha(b, torch.full((B,), seq[-1], dtype=torch.long, device=x.device))
        largest_sigmas = ((1 - largest_alphas) / largest_alphas).sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)

        inv_singulars = torch.zeros_like(Sigma)
        inv_singulars[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        init_y = torch.zeros(B, C * H * W, device=x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.shape)

        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars.view(1, -1) ** 2
        remaining_s = remaining_s.view(*x.shape).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        x = H_funcs.V(init_y.view(B, -1)).view(*x.shape)
        x = x / largest_sigmas

        for t_idx, t_val in enumerate(reversed(seq)):
            t = torch.full((B,), t_val, dtype=torch.long, device=x.device)
            alpha_t = compute_alpha(b, t)
            sigma_t = ((1 - alpha_t) / alpha_t).sqrt()[0, 0, 0, 0]

            # Step 1. Langevin refinement 먼저
            for _ in range(langevin_steps):
                epsilon = model(x, t)
                score = -epsilon / (1 - alpha_t).sqrt()
                noise = torch.randn_like(x)
                x = x + 0.5 * langevin_scale * score + (langevin_scale ** 0.5) * noise

            # Step 2. Refined x 기반 x0 추정
            epsilon = model(x, t)
            x0_t = (x - epsilon * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            # Step 3. Pseudo-inverse projection
            Vt_x0 = H_funcs.Vt(x0_t)
            Vt_x0 = Vt_x0 * Sigma
            Vt_x0[:, :U_t_y.shape[1]] = etaA * Sig_inv_U_t_y + (1 - etaA) * Vt_x0[:, :U_t_y.shape[1]]

            # Step 4. (Optional) Add noise in SVD space
            noise = langevin_scale * torch.randn_like(Vt_x0)
            Vt_x0 = Vt_x0 + noise

            # Step 5. Back to image space
            x = H_funcs.V(Vt_x0).view(*x.shape)
            x = alpha_t.sqrt() * x

    return x