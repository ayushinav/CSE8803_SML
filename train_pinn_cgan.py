import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import TensorDataset, DataLoader
import time
import datetime

# ============================================================
# Config
# ============================================================

DATA_FILE   = "data_mt.jld2"
G_SAVE_PATH = "G_with_pinn_16blocks_bs320.pt"
D_SAVE_PATH = "D_with_pinn_16blocks_bs320.pt"

z_dim       = 64
hidden_dim  = 512
num_blocks  = 16          # <-- depth 16
batch_size  = 128
num_epochs  = 200         
lr          = 1e-4
beta1       = 0.5
lambda_phy  = 1e-2        # physics loss weight
MU_0        = 4 * np.pi * 1e-7
EPS         = 1e-12


# ============================================================
# Utils
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Load dataset
# ============================================================

def load_mt_dataset(base_dir):
    path = os.path.join(base_dir, DATA_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"data file not found: {path}")

    with h5py.File(path, "r") as f:
        data_appres = np.array(f["data_appres"])   # (N, 61)
        data_phase  = np.array(f["data_phase"])    # (N, 61)
        x           = np.array(f["data_m"])        # (N, 50), we determined: log10(rho)
        omega_grid  = np.array(f["ω_grid"])        # (61,)
        mu_vec      = np.array(f["μ_vec"])         # unused

    y = np.concatenate([data_appres, data_phase], axis=1)  # (N, 122)

    x_mean = x.mean(axis=0, keepdims=True)
    x_std  = x.std(axis=0, keepdims=True) + 1e-8
    y_mean = y.mean(axis=0, keepdims=True)
    y_std  = y.std(axis=0, keepdims=True) + 1e-8

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    return {
        "x": x,
        "y": y,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "data_appres": data_appres,
        "data_phase": data_phase,
        "omega_grid": omega_grid,
        "mu_vec": mu_vec,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


# ============================================================
# 1D MT forward: batched, using rho (not sigma)
# ============================================================

def f_batch(rho_batch, h, omega, mu=MU_0):
    """
    rho_batch: (B, N_layers)
    h:        (N_layers,)
    omega:    (N_omega,)
    return:   (B, N_omega, 2)  [rho_a, phi]
    """
    device = rho_batch.device
    omega = omega.to(device=device, dtype=torch.float32)
    h     = h.to(device=device,     dtype=torch.float32)

    B, N_layers = rho_batch.shape
    N_omega     = omega.shape[0]

    rho_batch = torch.clamp(rho_batch, min=1e-4, max=1e4)

    rho_mat = rho_batch[:, None, :].expand(-1, N_omega, -1)   # (B, N_omega, N_layers)
    rho_c   = rho_mat.to(torch.complex64)

    omega_c = omega.to(torch.complex64)[None, :, None]        # (1, N_omega, 1)
    numerator = -1j * mu * omega_c                            # (1, N_omega, 1)

    k = torch.sqrt(numerator / (rho_c + EPS))                 # (B, N_omega, N_layers)

    R_next = None
    for i in range(N_layers - 2, -1, -1):
        k_i   = k[:, :, i]                                    # (B, N_omega)
        k_ip1 = k[:, :, i+1]                                  # (B, N_omega)
        denom = k_i + k_ip1
        denom = denom + EPS
        r_i   = (k_i - k_ip1) / denom
        h_i   = h[i]

        if R_next is None:
            R_current = r_i * torch.exp(-1j * k_i * h_i)
        else:
            h_ip1 = h[i+1]
            R_propagated = R_next * torch.exp(-1j * k_ip1 * h_ip1)
            denom2 = 1 + r_i * R_propagated
            denom2 = denom2 + EPS
            R_int  = (r_i + R_propagated) / denom2
            R_current = R_int * torch.exp(-1j * k_i * h_i)

        R_next = R_current

    R_surface = R_next                                        # (B, N_omega)

    k0 = k[:, :, 0]                                           # (B, N_omega)
    omega_c1 = omega.to(torch.complex64)[None, :]             # (1, N_omega)
    Z1 = (mu * omega_c1) / (k0 + EPS)                         # (B, N_omega)
    prop_factor_h0 = torch.exp(-1j * k0 * h[0])               # (B, N_omega)

    denomZ = R_surface * prop_factor_h0 - 1
    denomZ = denomZ + EPS
    Z = -Z1 * (R_surface * prop_factor_h0 + 1) / denomZ       # (B, N_omega)

    omega_b = omega[None, :]                                  # (1, N_omega)
    rho_a = (1.0 / (omega_b + EPS)) * (1.0 / mu) * torch.abs(Z)**2  # (B, N_omega)
    phi   = (180.0 / np.pi) * torch.atan2(Z.imag, Z.real)     # (B, N_omega)

    rho_a = torch.clamp(rho_a, 1e-2, 1e6)
    phi   = torch.clamp(phi, -90.0, 90.0)

    out = torch.stack((rho_a, phi), dim=2)                    # (B, N_omega, 2)
    out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    return out


def forward_mt_from_x_norm_batch(x_norm_batch,
                                 x_mean_t, x_std_t,
                                 h_t, omega_t,
                                 x_min_val, x_max_val):
    """
    x_norm_batch: (B, 50)  -- normalized log10(rho)
    returns y_pred_phys: (B, 122)  -- [rho_a, phi] in physical scale
    """
    # back to log10(rho)
    x_log_rho = x_norm_batch * x_std_t + x_mean_t            # (B, 50)

    # clamp log10(rho) to slightly extended data range
    x_log_rho = torch.clamp(
        x_log_rho,
        x_min_val - 0.5,
        x_max_val + 0.5
    )

    rho_layers = torch.pow(10.0, x_log_rho)                  # (B, 50)

    out = f_batch(rho_layers, h_t, omega_t)                  # (B, 61, 2)
    rho_a = out[:, :, 0]                                     # (B, 61)
    phi   = out[:, :, 1]                                     # (B, 61)

    y_pred_phys = torch.cat([rho_a, phi], dim=1)             # (B, 122)
    y_pred_phys = torch.nan_to_num(y_pred_phys, nan=0.0, posinf=1e6, neginf=-1e6)
    return y_pred_phys


# ============================================================
# GAN networks: depth 16
# ============================================================

class ResBlockG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out


class ResBlockD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn_utils.spectral_norm(nn.Linear(dim, dim))
        self.fc2 = nn_utils.spectral_norm(nn.Linear(dim, dim))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = self.act(out + residual)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, x_dim, hidden=512, num_blocks=16):
        super().__init__()
        in_dim = z_dim + y_dim
        self.fc_in = nn.Linear(in_dim, hidden)
        self.bn_in = nn.BatchNorm1d(hidden)
        self.act   = nn.LeakyReLU(0.2)
        blocks = [ResBlockG(hidden) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden, x_dim)

    def forward(self, z, y):
        inp = torch.cat([z, y], dim=1)
        h = self.fc_in(inp)
        h = self.bn_in(h)
        h = self.act(h)
        h = self.blocks(h)
        x_fake = self.fc_out(h)
        return x_fake


class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim, hidden=512, num_blocks=16):
        super().__init__()
        in_dim = x_dim + y_dim
        self.fc_in = nn_utils.spectral_norm(nn.Linear(in_dim, hidden))
        self.act   = nn.LeakyReLU(0.2)
        blocks = [ResBlockD(hidden) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.fc_out = nn_utils.spectral_norm(nn.Linear(hidden, 1))

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        h = self.act(self.fc_in(inp))
        h = self.blocks(h)
        logit = self.fc_out(h)
        return logit


# ============================================================
# Training
# ============================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = get_device()
    print("Device:", device)

    # load data
    data = load_mt_dataset(base_dir)
    x        = data["x"]         # (N, 50), log10(rho)
    y        = data["y"]         # (N, 122)
    x_norm   = data["x_norm"]
    y_norm   = data["y_norm"]
    x_mean   = data["x_mean"]
    x_std    = data["x_std"]
    y_mean   = data["y_mean"]
    y_std    = data["y_std"]
    omega    = data["omega_grid"]

    x_dim = x.shape[1]
    y_dim = y.shape[1]
    N     = x.shape[0]

    print(f"N = {N}, x_dim = {x_dim}, y_dim = {y_dim}")

    # thickness: simple uniform here
    h_np = np.ones(x_dim, dtype=np.float32) * 100.0

    # tensors on device
    x_mean_t = torch.from_numpy(x_mean).float().to(device)
    x_std_t  = torch.from_numpy(x_std ).float().to(device)
    y_mean_t = torch.from_numpy(y_mean).float().to(device)
    y_std_t  = torch.from_numpy(y_std ).float().to(device)
    omega_t  = torch.from_numpy(omega).float().to(device)
    h_t      = torch.from_numpy(h_np).float().to(device)

    # x range for clamping
    x_min_val = float(x.min())
    x_max_val = float(x.max())
    print(f"x range in data (log10 rho): [{x_min_val:.3f}, {x_max_val:.3f}]")

    # dataloader (on normalized x,y)
    x_tensor = torch.from_numpy(x_norm).float()
    y_tensor = torch.from_numpy(y_norm).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # build models
    G = Generator(z_dim, y_dim, x_dim, hidden=hidden_dim, num_blocks=num_blocks).to(device)
    D = Discriminator(x_dim, y_dim, hidden=hidden_dim, num_blocks=num_blocks).to(device)

    print(G)
    print(D)

    criterion_adv = nn.BCEWithLogitsLoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    # training loop
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        for step, (x_real, y_cond) in enumerate(dataloader, start=1):
            x_real = x_real.to(device)   # (B, 50) normalized
            y_cond = y_cond.to(device)   # (B, 122) normalized
            batch_size_cur = x_real.size(0)

            real_labels = torch.ones(batch_size_cur, 1, device=device)
            fake_labels = torch.zeros(batch_size_cur, 1, device=device)

            # ---- update D ----
            optimizer_D.zero_grad()

            out_real = D(x_real, y_cond)
            loss_D_real = criterion_adv(out_real, real_labels)

            z = torch.randn(batch_size_cur, z_dim, device=device)
            with torch.no_grad():
                x_fake_for_D = G(z, y_cond)
            out_fake = D(x_fake_for_D, y_cond)
            loss_D_fake = criterion_adv(out_fake, fake_labels)

            loss_D = loss_D_real + loss_D_fake

            if torch.isfinite(loss_D).all():
                loss_D.backward()
                nn.utils.clip_grad_norm_(D.parameters(), max_norm=10.0)
                optimizer_D.step()
            else:
                print("NaN/Inf in D loss, skip batch")

            # ---- update G (adv + physics) ----
            optimizer_G.zero_grad()

            z = torch.randn(batch_size_cur, z_dim, device=device)
            x_fake = G(z, y_cond)  # normalized x

            out_fake_for_G = D(x_fake, y_cond)
            loss_G_adv = criterion_adv(out_fake_for_G, real_labels)

            # physics loss: compare in physical y-space (un-normalized)
            y_obs_phys = y_cond * y_std_t + y_mean_t               # (B, 122)

            y_pred_phys = forward_mt_from_x_norm_batch(
                x_fake, x_mean_t, x_std_t, h_t, omega_t,
                x_min_val, x_max_val
            )

            loss_G_phy = F.mse_loss(y_pred_phys, y_obs_phys)
            loss_G = loss_G_adv + lambda_phy * loss_G_phy

            if torch.isfinite(loss_G).all():
                loss_G.backward()
                nn.utils.clip_grad_norm_(G.parameters(), max_norm=10.0)
                optimizer_G.step()
            else:
                print("NaN/Inf in G loss, skip batch (G)")

            if step % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"Step {step}/{len(dataloader)} "
                    f"D: {loss_D.item():.4f} "
                    f"G_adv: {loss_G_adv.item():.4f} "
                    f"G_phy: {loss_G_phy.item():.4f}"
                )
        epoch_time = time.perf_counter() - epoch_start
        print(f"==> Epoch {epoch+1} done (time {epoch_time:.2f}s)")
    total_time = time.perf_counter() - start_time

    print("Training done. Saving models...")
    print(f"Total training time: {total_time:.2f} seconds ({str(datetime.timedelta(seconds=int(total_time)))})")

    torch.save(G.state_dict(), os.path.join(base_dir, G_SAVE_PATH))
    torch.save(D.state_dict(), os.path.join(base_dir, D_SAVE_PATH))

    print("Saved:")
    print("  Generator ->", os.path.join(base_dir, G_SAVE_PATH))
    print("  Discriminator ->", os.path.join(base_dir, D_SAVE_PATH))


if __name__ == "__main__":
    main()
