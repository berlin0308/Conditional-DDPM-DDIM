import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms



class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):

        """
        Training
        """
        # Step 3: t ~ Uniform({1,...,T})
        t = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)

        # Step 4: eps ~ N(0, I), noise
        eps = torch.randn_like(x)

        # Step 5-1: Xt
        x_t = (
            self.sqrtab[t, None, None, None] * x
            + self.sqrtmab[t, None, None, None] * eps
        )  

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        # Step 5-2: predicted noise
        eps_pred = self.nn_model(x_t, c, t / self.n_T, context_mask)

        # Step 5-3: compute loss
        return self.loss_mse(eps, eps_pred)


    def sample(self, n_sample, size, device, guide_w=0.0):

        """
        Sampling
        """
        # Step 1: XT ~ N(0, I), initial noise
        Xt = torch.randn(n_sample, *size).to(device)

        c_i = torch.arange(0, 10).to(device)
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))
        context_mask = torch.zeros_like(c_i).to(device)
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0
        Xt_store = []

        # Step 2: t = T,T-1,...,1
        for i in range(self.n_T, 0, -1):
            print(f"sampling timestep {i}", end="\r")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            Xt = Xt.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            # Step 3: z ~ N(0, I)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # Step 4-1: predicted noise
            eps = self.nn_model(Xt, c_i, t_is, context_mask)

            # Step 4-2: X(t-1)
            eps1 = eps[:n_sample]  # with condition
            eps2 = eps[n_sample:]  # without condition
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            Xt = Xt[:n_sample]
            Xt = (
                self.oneover_sqrta[i] * (Xt - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                Xt_store.append(Xt.detach().cpu().numpy())

        Xt_store = np.array(Xt_store)

        return Xt, Xt_store


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        """
        process and downscale the image feature maps
        """
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        """
        process and upscale the image feature maps
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        generic one layer FC NN for embedding things  
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()

        # %% 參數設定
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(
                2 * n_feat, 2 * n_feat, 7, 7
            ),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    # self.nn_model(x_t, c, _ts / self.n_T, context_mask)
    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = -1 * (1 - context_mask)  # need to flip 0 <-> 1
        c = c * context_mask  # 決定是否要有condition

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class CombinedDDPM(nn.Module):
    def __init__(self, ddpm1, ddpm2):
        super(CombinedDDPM, self).__init__()
        self.ddpm1 = ddpm1
        self.ddpm2 = ddpm2

    def forward(self, x, c, mode=1):
        if mode == 1:
            return self.ddpm1(x, c)
        elif mode == 2:
            return self.ddpm2(x, c)
        else:
            raise ValueError("Invalid mode. Mode should be 1 or 2.")

    def sample(self, n_sample, size, device, mode=1, guide_w=0.0):
        if mode == 1:
            return self.ddpm1.sample(n_sample, size, device, guide_w)
        elif mode == 2:
            return self.ddpm2.sample(n_sample, size, device, guide_w)
        else:
            raise ValueError("Invalid mode. Mode should be 1 or 2.")

if __name__ == "__main__":

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    ddpm1 = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=500,
        device=device,
        drop_prob=0.0,
    )

    ddpm2 = ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=128, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=500,
        device=device,
        drop_prob=0.0
    )

    ddpm1.load_state_dict(torch.load("ddpm_mnistm_ep99.pth", map_location=device))
    ddpm2.load_state_dict(torch.load("ddpm_svhn_ep99.pth", map_location=device))

    combined_ddpm = CombinedDDPM(ddpm1=ddpm1, ddpm2=ddpm2)
    torch.save(combined_ddpm.state_dict(), "combined_ddpm.pth")

    combined_ddpm_loaded = CombinedDDPM(ddpm1=ddpm1, ddpm2=ddpm2)
    combined_ddpm_loaded.load_state_dict(torch.load("combined_ddpm.pth", map_location=device))



