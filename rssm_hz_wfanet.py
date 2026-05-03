import torch
import torch.nn as nn
import torch.nn.functional as F

from net_torch import DWT_2D, IDWT_2D, raise_channel, reduce_channel, resblock


class WaveletPyramid(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        self.dwt = DWT_2D()

    def forward(self, x):
        coeffs = []
        current = x
        for _ in range(self.levels):
            dec = self.dwt(current)
            c = dec.shape[1] // 4
            ll = dec[:, :c]
            lh = dec[:, c: 2 * c]
            hl = dec[:, 2 * c: 3 * c]
            hh = dec[:, 3 * c:]
            coeffs.append((ll, lh, hl, hh))
            current = ll
        return coeffs


class WaveletReconstructor(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        self.idwt = IDWT_2D()

    def forward(self, coeffs):
        current = coeffs[-1][0]
        for i in range(self.levels - 1, -1, -1):
            ll, lh, hl, hh = coeffs[i]
            if i == self.levels - 1:
                pack = torch.cat([ll, lh, hl, hh], dim=1)
            else:
                pack = torch.cat([current, lh, hl, hh], dim=1)
            current = self.idwt(pack)
        return current


class ConvGRUCell2d(nn.Module):
    """2D convolutional GRU cell that preserves spatial structure during state updates."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.reset = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.update = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        r = torch.sigmoid(self.reset(combined))
        u = torch.sigmoid(self.update(combined))
        n = torch.tanh(self.candidate(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - u) * h_prev + u * n


class RSSMHzCell(nn.Module):
    def __init__(self, obs_dim, hidden_dim, latent_dim, deterministic_only=False, use_conv_gru=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru

        if use_conv_gru:
            self.gru = ConvGRUCell2d(obs_dim + latent_dim, hidden_dim, kernel_size=3)
            self.prior = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, latent_dim * 2, 1),
            )
            self.posterior = nn.Sequential(
                nn.Conv2d(hidden_dim + obs_dim, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, latent_dim * 2, 1),
            )
        else:
            self.gru = nn.GRUCell(obs_dim + latent_dim, hidden_dim)
            self.prior = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )
            self.posterior = nn.Sequential(
                nn.Linear(hidden_dim + obs_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )

    @staticmethod
    def _sample(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return torch.clamp(z, min=-6.0, max=6.0)

    @staticmethod
    def _kl_div(mu_q, logvar_q, mu_p, logvar_p):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0)
        # Sum over channel dim (dim=1) for 2D, same for 1D
        return kl.sum(dim=1)

    def _forward_2d(self, obs, h_prev, z_prev, training):
        h_bar = self.gru(torch.cat([obs, z_prev], dim=1), h_prev)
        if self.deterministic_only:
            z = torch.zeros_like(z_prev)
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
            return h_bar, z, kl

        prior_stats = self.prior(h_bar)
        mu_p, logvar_p = torch.chunk(prior_stats, 2, dim=1)
        logvar_p = torch.clamp(logvar_p, min=-8.0, max=2.0)

        if training:
            post_stats = self.posterior(torch.cat([h_bar, obs], dim=1))
            mu_q, logvar_q = torch.chunk(post_stats, 2, dim=1)
            logvar_q = torch.clamp(logvar_q, min=-8.0, max=2.0)
            z = self._sample(mu_q, logvar_q)
            kl = self._kl_div(mu_q, logvar_q, mu_p, logvar_p)
        else:
            z = mu_p
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        return h_bar, z, kl

    def _forward_1d(self, obs, h_prev, z_prev, training):
        h_bar = self.gru(torch.cat([obs, z_prev], dim=1), h_prev)
        if self.deterministic_only:
            z = torch.zeros_like(z_prev)
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
            return h_bar, z, kl

        prior_stats = self.prior(h_bar)
        mu_p, logvar_p = torch.chunk(prior_stats, 2, dim=1)
        logvar_p = torch.clamp(logvar_p, min=-8.0, max=2.0)

        if training:
            post_stats = self.posterior(torch.cat([h_bar, obs], dim=1))
            mu_q, logvar_q = torch.chunk(post_stats, 2, dim=1)
            logvar_q = torch.clamp(logvar_q, min=-8.0, max=2.0)
            z = self._sample(mu_q, logvar_q)
            kl = self._kl_div(mu_q, logvar_q, mu_p, logvar_p)
        else:
            z = mu_p
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        return h_bar, z, kl

    def forward(self, obs, h_prev, z_prev, training=True):
        if self.use_conv_gru:
            return self._forward_2d(obs, h_prev, z_prev, training)
        else:
            return self._forward_1d(obs, h_prev, z_prev, training)


class CrossScaleFusionHz(nn.Module):
    def __init__(self, pan_channels, ms_channels, hidden_dim, latent_dim, deterministic_only=False, use_conv_gru=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru

        self.pan_proj = nn.Sequential(
            nn.Conv2d(pan_channels, hidden_dim // 2, 1, bias=True),
            nn.PReLU(hidden_dim // 2),
        )
        self.ms_proj = nn.Sequential(
            nn.Conv2d(ms_channels, hidden_dim // 2, 1, bias=True),
            nn.PReLU(hidden_dim // 2),
        )

        self.cell = RSSMHzCell(hidden_dim, hidden_dim, latent_dim,
                               deterministic_only=deterministic_only, use_conv_gru=use_conv_gru)

        self.hz_to_feat = nn.Sequential(
            nn.Conv2d(hidden_dim + latent_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.PReLU(hidden_dim),
            nn.Conv2d(hidden_dim, ms_channels, 3, 1, 1, bias=True),
        )

        self.obs_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, pan_feat, ms_feat, h_prev, z_prev, training=True):
        p = self.pan_proj(pan_feat)
        m = self.ms_proj(ms_feat)
        obs = torch.cat([p, m], dim=1)

        if self.use_conv_gru:
            h_state, z_state, kl = self.cell(obs, h_prev, z_prev, training=training)
        else:
            b, _, h, w = pan_feat.shape
            obs_flat = obs.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim)
            h_prev_flat = h_prev.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim)
            z_prev_flat = z_prev.permute(0, 2, 3, 1).reshape(b * h * w, self.latent_dim)
            h_flat, z_flat, kl = self.cell(obs_flat, h_prev_flat, z_prev_flat, training=training)
            h_state = h_flat.reshape(b, h, w, self.hidden_dim).permute(0, 3, 1, 2)
            z_state = z_flat.reshape(b, h, w, self.latent_dim).permute(0, 3, 1, 2)

        fused_raw = self.hz_to_feat(torch.cat([h_state, z_state], dim=1))
        gate = self.obs_gate(obs)
        fused = fused_raw * gate + ms_feat

        kl_mean = kl.mean() if kl.numel() > 0 else kl
        return fused, h_state, z_state, kl_mean


class RSSMWaveletFusionHz(nn.Module):
    def __init__(self, pan_channels_per_level, ms_channels_per_level, hidden_dim=96, latent_dim=32, levels=3,
                 deterministic_only=False, separate_subband_gates=True, use_conv_gru=False,
                 learnable_fusion=False):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru
        self.learnable_fusion = learnable_fusion

        self.fusion_blocks = nn.ModuleList(
            [
                CrossScaleFusionHz(
                    pan_channels=pan_channels_per_level[i],
                    ms_channels=ms_channels_per_level[i],
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    deterministic_only=deterministic_only,
                    use_conv_gru=use_conv_gru,
                )
                for i in range(levels)
            ]
        )

        self.state_up_h = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1, bias=True),
                    nn.PReLU(hidden_dim),
                )
                for _ in range(levels - 1)
            ]
        )
        self.state_up_z = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(latent_dim, latent_dim, 4, 2, 1, bias=True),
                    nn.PReLU(latent_dim),
                )
                for _ in range(levels - 1)
            ]
        )

        self.separate_subband_gates = separate_subband_gates

        self.pan_high_to_ms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "lh": nn.Conv2d(pan_channels_per_level[i] // 4, ms_channels_per_level[i], 1, bias=True),
                        "hl": nn.Conv2d(pan_channels_per_level[i] // 4, ms_channels_per_level[i], 1, bias=True),
                        "hh": nn.Conv2d(pan_channels_per_level[i] // 4, ms_channels_per_level[i], 1, bias=True),
                    }
                )
                for i in range(levels)
            ]
        )

        # Project z_state (latent_dim) to MS channel space so it can inform the high-frequency gates.
        self.z_to_gate = nn.ModuleList(
            [
                nn.Conv2d(latent_dim, ms_channels_per_level[i], 1, bias=True)
                for i in range(levels)
            ]
        )

        # Gate input = [fused_ll, ll_ms, pan_lh, pan_hl, pan_hh, z_gate] -> 6 * ms_channels
        gate_in_channels = [ms_channels_per_level[i] * 6 for i in range(levels)]
        if separate_subband_gates:
            self.high_gate_lh = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )
            self.high_gate_hl = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )
            self.high_gate_hh = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )
        else:
            self.high_gate = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(gate_in_channels[i], ms_channels_per_level[i], 1, bias=True), nn.Sigmoid())
                 for i in range(levels)]
            )

        # Learnable fusion: replace additive injection with small ConvNets.
        # Each ConvFusion takes [MS_hf, PAN_hf, alpha * PAN_hf] and produces fused output.
        if learnable_fusion:
            def _make_conv_fusion(in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=True),
                    nn.PReLU(out_ch),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
                    nn.PReLU(out_ch),
                    nn.Conv2d(out_ch, out_ch, 1, bias=True),
                )
            # Input: MS_hf (ms_c) + PAN_hf (ms_c) + alpha*PAN_hf (ms_c) = 3*ms_c
            fusion_in_channels = [ms_channels_per_level[i] * 3 for i in range(levels)]
            self.conv_fusion_lh = nn.ModuleList(
                [_make_conv_fusion(fusion_in_channels[i], ms_channels_per_level[i]) for i in range(levels)]
            )
            self.conv_fusion_hl = nn.ModuleList(
                [_make_conv_fusion(fusion_in_channels[i], ms_channels_per_level[i]) for i in range(levels)]
            )
            self.conv_fusion_hh = nn.ModuleList(
                [_make_conv_fusion(fusion_in_channels[i], ms_channels_per_level[i]) for i in range(levels)]
            )

        self.wavelet_reconstructor = WaveletReconstructor(levels=levels)
        self.refine = nn.Sequential(resblock(ms_channels_per_level[0]), resblock(ms_channels_per_level[0]))

    def forward(self, pan_pyramid, ms_pyramid, training=True):
        h_state = None
        z_state = None
        fused_coeffs = []
        kl_terms = []

        for level in range(self.levels - 1, -1, -1):
            ll_pan, lh_pan, hl_pan, hh_pan = pan_pyramid[level]
            ll_ms, lh_ms, hl_ms, hh_ms = ms_pyramid[level]
            b, _, h, w = ll_pan.shape

            pan_feat = torch.cat([ll_pan, lh_pan, hl_pan, hh_pan], dim=1)
            ms_feat = ll_ms

            if h_state is None:
                h_state = torch.zeros(b, self.hidden_dim, h, w, device=ll_pan.device, dtype=ll_pan.dtype)
                z_state = torch.zeros(b, self.latent_dim, h, w, device=ll_pan.device, dtype=ll_pan.dtype)
            else:
                h_state = self.state_up_h[level](h_state)
                z_state = self.state_up_z[level](z_state)
                if h_state.shape[2:] != (h, w):
                    h_state = F.interpolate(h_state, size=(h, w), mode="bilinear", align_corners=False)
                if z_state.shape[2:] != (h, w):
                    z_state = F.interpolate(z_state, size=(h, w), mode="bilinear", align_corners=False)

            fused_ll, h_state, z_state, kl_mean = self.fusion_blocks[level](
                pan_feat, ms_feat, h_state, z_state, training=training
            )
            kl_terms.append(kl_mean)

            pan_lh = self.pan_high_to_ms[level]["lh"](lh_pan)
            pan_hl = self.pan_high_to_ms[level]["hl"](hl_pan)
            pan_hh = self.pan_high_to_ms[level]["hh"](hh_pan)

            z_gate = self.z_to_gate[level](z_state)
            gate_in = torch.cat([fused_ll, ll_ms, pan_lh, pan_hl, pan_hh, z_gate], dim=1)

            if self.separate_subband_gates:
                alpha_lh = self.high_gate_lh[level](gate_in)
                alpha_hl = self.high_gate_hl[level](gate_in)
                alpha_hh = self.high_gate_hh[level](gate_in)
                if self.learnable_fusion:
                    fused_lh = self.conv_fusion_lh[level](torch.cat([lh_ms, pan_lh, alpha_lh * pan_lh], dim=1))
                    fused_hl = self.conv_fusion_hl[level](torch.cat([hl_ms, pan_hl, alpha_hl * pan_hl], dim=1))
                    fused_hh = self.conv_fusion_hh[level](torch.cat([hh_ms, pan_hh, alpha_hh * pan_hh], dim=1))
                else:
                    fused_lh = lh_ms + alpha_lh * pan_lh
                    fused_hl = hl_ms + alpha_hl * pan_hl
                    fused_hh = hh_ms + alpha_hh * pan_hh
            else:
                alpha = self.high_gate[level](gate_in)
                if self.learnable_fusion:
                    fused_lh = self.conv_fusion_lh[level](torch.cat([lh_ms, pan_lh, alpha * pan_lh], dim=1))
                    fused_hl = self.conv_fusion_hl[level](torch.cat([hl_ms, pan_hl, alpha * pan_hl], dim=1))
                    fused_hh = self.conv_fusion_hh[level](torch.cat([hh_ms, pan_hh, alpha * pan_hh], dim=1))
                else:
                    fused_lh = lh_ms + alpha * pan_lh
                    fused_hl = hl_ms + alpha * pan_hl
                    fused_hh = hh_ms + alpha * pan_hh

            fused_coeffs.append((fused_ll, fused_lh, fused_hl, fused_hh))

        fused_coeffs = list(reversed(fused_coeffs))
        recon = self.wavelet_reconstructor(fused_coeffs)
        recon = self.refine(recon)

        kl_loss = torch.stack(kl_terms).mean() if kl_terms else recon.new_tensor(0.0)
        return recon, kl_loss


class RSSMHWViTHZ(nn.Module):
    def __init__(
        self,
        L_up_channel,
        pan_channel,
        pan_target_channel,
        ms_target_channel,
        hidden_dim=96,
        latent_dim=32,
        deterministic_only=False,
        separate_subband_gates=True,
        use_conv_gru=False,
        learnable_fusion=False,
        image_space_wavelet=False,
    ):
        super().__init__()
        self.deterministic_only = deterministic_only
        self.image_space_wavelet = image_space_wavelet

        self.pan_raise = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
        self.ms_upsample = nn.Sequential(
            nn.Conv2d(L_up_channel, L_up_channel * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        self.ms_act = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.ms_raise = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)

        self.wavelet = WaveletPyramid(levels=3)

        if image_space_wavelet:
            # Wavelet on raw images: PAN(1-ch) and MS_up(L_up_channel-ch).
            # Each PAN subband (1-ch) is raised to pan_target_channel via a shared raiser.
            self.pan_subband_raise = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
            # Each MS subband (L_up_channel-ch) is raised to ms_target_channel via a shared raiser.
            self.ms_subband_raise = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)

        pan_channels_per_level = [pan_target_channel * 4] * 3
        ms_channels_per_level = [ms_target_channel] * 3

        self.rssm_fusion = RSSMWaveletFusionHz(
            pan_channels_per_level=pan_channels_per_level,
            ms_channels_per_level=ms_channels_per_level,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            levels=3,
            deterministic_only=deterministic_only,
            separate_subband_gates=separate_subband_gates,
            use_conv_gru=use_conv_gru,
            learnable_fusion=learnable_fusion,
        )

        self.reduce = reduce_channel(ms_target_channel=ms_target_channel, L_up_channel=L_up_channel)
        self.out_act = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.fused_weight = nn.Parameter(torch.ones(1))  # default 1.0 = original behavior

    def forward(self, pan, ms, lms):
        ms_up = self.ms_upsample(ms)
        ms_up = self.ms_act(ms_up + lms)

        if self.image_space_wavelet:
            # Wavelet decompose raw PAN and MS_up in image space.
            pan_pyr_raw = self.wavelet(pan)
            ms_pyr_raw = self.wavelet(ms_up)

            # Raise each subband from image-space channels to feature-space channels.
            pan_pyr = []
            ms_pyr = []
            for level in range(3):
                ll_p, lh_p, hl_p, hh_p = pan_pyr_raw[level]   # 1-ch each
                ll_m, lh_m, hl_m, hh_m = ms_pyr_raw[level]   # L_up_channel-ch each

                ll_p_r = self.pan_subband_raise(ll_p)
                lh_p_r = self.pan_subband_raise(lh_p)
                hl_p_r = self.pan_subband_raise(hl_p)
                hh_p_r = self.pan_subband_raise(hh_p)

                ll_m_r = self.ms_subband_raise(ll_m)
                lh_m_r = self.ms_subband_raise(lh_m)
                hl_m_r = self.ms_subband_raise(hl_m)
                hh_m_r = self.ms_subband_raise(hh_m)

                pan_pyr.append((ll_p_r, lh_p_r, hl_p_r, hh_p_r))
                ms_pyr.append((ll_m_r, lh_m_r, hl_m_r, hh_m_r))
        else:
            pan_feat = self.pan_raise(pan)
            ms_feat = self.ms_raise(ms_up)
            pan_pyr = self.wavelet(pan_feat)
            ms_pyr = self.wavelet(ms_feat)

        fused, kl_loss = self.rssm_fusion(pan_pyr, ms_pyr, training=self.training)
        fused = self.reduce(fused)
        out = self.out_act(self.fused_weight * fused + ms_up)
        return out, kl_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSSMHWViTHZ(
        L_up_channel=8,
        pan_channel=1,
        pan_target_channel=32,
        ms_target_channel=32,
        hidden_dim=96,
        latent_dim=32,
    ).to(device)
    pan = torch.randn(2, 1, 64, 64, device=device)
    ms = torch.randn(2, 8, 16, 16, device=device)
    lms = torch.randn(2, 8, 64, 64, device=device)
    out, kl = model(pan, ms, lms)
    print(out.shape, kl.item())
