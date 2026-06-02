import torch
import torch.nn as nn
import torch.nn.functional as F

from net_torch import DWT_2D, IDWT_2D, raise_channel, reduce_channel, resblock

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None


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


class ZeroInitResidual(nn.Module):
    """Small residual block whose last layer starts from zero output."""
    def __init__(self, channels, hidden_channels=None, depthwise=True):
        super().__init__()
        hidden_channels = hidden_channels or channels
        groups = channels if depthwise else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


class StateSpatialMixer(nn.Module):
    """Local spatial context mixer for recurrent hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mix = ZeroInitResidual(hidden_dim, hidden_channels=hidden_dim, depthwise=True)

    def forward(self, h_state):
        return self.mix(h_state)


class RSSMHzCell(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        latent_dim,
        deterministic_only=False,
        use_conv_gru=False,
        z_eval_mode="prior",
        z_update_order="legacy",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru
        self.z_eval_mode = z_eval_mode
        self.z_update_order = z_update_order

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

    def set_z_eval_mode(self, mode):
        if mode not in {"prior", "posterior", "zero"}:
            raise ValueError(f"Unsupported z_eval_mode: {mode}")
        self.z_eval_mode = mode

    def _latent_from_state(self, h_ref, obs, z_like, training, force_zero_z=False):
        if self.deterministic_only or force_zero_z:
            z = torch.zeros_like(z_like)
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
            return z, kl

        prior_stats = self.prior(h_ref)
        mu_p, logvar_p = torch.chunk(prior_stats, 2, dim=1)
        logvar_p = torch.clamp(logvar_p, min=-8.0, max=2.0)

        need_posterior = training or self.z_eval_mode == "posterior"
        if need_posterior:
            post_stats = self.posterior(torch.cat([h_ref, obs], dim=1))
            mu_q, logvar_q = torch.chunk(post_stats, 2, dim=1)
            logvar_q = torch.clamp(logvar_q, min=-8.0, max=2.0)
            kl = self._kl_div(mu_q, logvar_q, mu_p, logvar_p)
            z = self._sample(mu_q, logvar_q) if training else mu_q
        elif self.z_eval_mode == "zero":
            z = torch.zeros_like(mu_p)
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        else:
            z = mu_p
            kl = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        return z, kl

    def _forward_2d(self, obs, h_prev, z_prev, training, force_zero_z=False):
        if self.z_update_order == "innovation":
            z, kl = self._latent_from_state(h_prev, obs, z_prev, training, force_zero_z=force_zero_z)
            h_state = self.gru(torch.cat([obs, z], dim=1), h_prev)
            return h_state, z, kl

        h_bar = self.gru(torch.cat([obs, z_prev], dim=1), h_prev)
        z, kl = self._latent_from_state(h_bar, obs, z_prev, training, force_zero_z=force_zero_z)
        return h_bar, z, kl

    def _forward_1d(self, obs, h_prev, z_prev, training, force_zero_z=False):
        if self.z_update_order == "innovation":
            z, kl = self._latent_from_state(h_prev, obs, z_prev, training, force_zero_z=force_zero_z)
            h_state = self.gru(torch.cat([obs, z], dim=1), h_prev)
            return h_state, z, kl

        h_bar = self.gru(torch.cat([obs, z_prev], dim=1), h_prev)
        z, kl = self._latent_from_state(h_bar, obs, z_prev, training, force_zero_z=force_zero_z)
        return h_bar, z, kl

    def forward(self, obs, h_prev, z_prev, training=True, force_zero_z=False):
        if self.use_conv_gru:
            return self._forward_2d(obs, h_prev, z_prev, training, force_zero_z=force_zero_z)
        else:
            return self._forward_1d(obs, h_prev, z_prev, training, force_zero_z=force_zero_z)


class LevelLLCorrection(nn.Module):
    """Level-wise LL correction before IDWT reconstruction.

    This is more targeted than the final-image low-frequency head: it lets each
    wavelet level correct LL/spectral bias while preserving the MS_LL residual
    path at initialization.
    """
    def __init__(self, pan_channels, ms_channels):
        super().__init__()
        hidden = ms_channels
        self.body = nn.Sequential(
            nn.Conv2d(ms_channels * 2 + pan_channels, hidden, 1, 1, 0, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, ms_channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, fused_ll, ms_ll, pan_feat):
        delta = self.body(torch.cat([fused_ll, ms_ll, pan_feat], dim=1))
        return fused_ll + delta


class CrossScaleFusionHz(nn.Module):
    def __init__(
        self,
        pan_channels,
        ms_channels,
        hidden_dim,
        latent_dim,
        deterministic_only=False,
        use_conv_gru=False,
        use_state_spatial_mixer=False,
        use_level_ll_corr=False,
        z_eval_mode="prior",
        z_update_order="legacy",
    ):
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
                               deterministic_only=deterministic_only, use_conv_gru=use_conv_gru,
                               z_eval_mode=z_eval_mode, z_update_order=z_update_order)
        self.state_mixer = StateSpatialMixer(hidden_dim) if use_state_spatial_mixer else None

        self.hz_to_feat = nn.Sequential(
            nn.Conv2d(hidden_dim + latent_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.PReLU(hidden_dim),
            nn.Conv2d(hidden_dim, ms_channels, 3, 1, 1, bias=True),
        )

        self.obs_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.level_ll_corr = LevelLLCorrection(pan_channels, ms_channels) if use_level_ll_corr else None

    def set_z_eval_mode(self, mode):
        self.cell.set_z_eval_mode(mode)

    def _decode_ll(self, h_state, z_state, obs, ms_feat, pan_feat):
        fused_raw = self.hz_to_feat(torch.cat([h_state, z_state], dim=1))
        gate = self.obs_gate(obs)
        fused = fused_raw * gate + ms_feat
        if self.level_ll_corr is not None:
            fused = self.level_ll_corr(fused, ms_feat, pan_feat)
        return fused

    def forward(self, pan_feat, ms_feat, h_prev, z_prev, training=True, force_zero_z=False, return_z0=False):
        p = self.pan_proj(pan_feat)
        m = self.ms_proj(ms_feat)
        obs = torch.cat([p, m], dim=1)

        if self.use_conv_gru:
            h_state, z_state, kl = self.cell(obs, h_prev, z_prev, training=training, force_zero_z=force_zero_z)
        else:
            b, _, h, w = pan_feat.shape
            obs_flat = obs.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim)
            h_prev_flat = h_prev.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim)
            z_prev_flat = z_prev.permute(0, 2, 3, 1).reshape(b * h * w, self.latent_dim)
            h_flat, z_flat, kl = self.cell(
                obs_flat,
                h_prev_flat,
                z_prev_flat,
                training=training,
                force_zero_z=force_zero_z,
            )
            h_state = h_flat.reshape(b, h, w, self.hidden_dim).permute(0, 3, 1, 2)
            z_state = z_flat.reshape(b, h, w, self.latent_dim).permute(0, 3, 1, 2)

        if self.state_mixer is not None:
            h_state = self.state_mixer(h_state)

        fused = self._decode_ll(h_state, z_state, obs, ms_feat, pan_feat)
        fused_z0 = None
        if return_z0:
            fused_z0 = self._decode_ll(h_state, torch.zeros_like(z_state), obs, ms_feat, pan_feat)

        kl_mean = kl.mean() if kl.numel() > 0 else kl
        return fused, h_state, z_state, kl_mean, fused_z0


class LowFreqCorrection(nn.Module):
    """Lightweight low-frequency/spectral correction head.

    The last conv is zero-initialized, so enabling this module starts from the
    original RSSM-HZ output and learns only a residual correction when useful.
    """
    def __init__(self, channels, hidden_channels=32, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.body = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, fused, ms_up, pan):
        pad = self.kernel_size // 2
        fused_lf = F.avg_pool2d(fused, self.kernel_size, stride=1, padding=pad)
        ms_lf = F.avg_pool2d(ms_up, self.kernel_size, stride=1, padding=pad)
        pan_lf = F.avg_pool2d(pan, self.kernel_size, stride=1, padding=pad)
        return self.gamma * self.body(torch.cat([fused_lf, ms_lf, pan_lf], dim=1))


class BandAwareCorrection(nn.Module):
    """Per-band spectral residual correction with local and global cues.

    GF2/QB error analysis showed that the remaining error is concentrated in a
    few multispectral bands. This head is deliberately small and zero-initialized:
    it starts as an exact no-op and learns only a residual correction on top of
    the existing RSSM-HZ output. Its cost is linear in the number of pixels.
    """
    def __init__(self, channels, hidden_channels=32, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        in_channels = channels * 3 + 1
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, groups=hidden_channels, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )
        self.global_affine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels * 2, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.global_affine[-1].weight)
        nn.init.zeros_(self.global_affine[-1].bias)

    def forward(self, base, fused, ms_up, pan):
        x = torch.cat([base, fused, ms_up, pan], dim=1)
        local_delta = self.local(x)
        stats = F.adaptive_avg_pool2d(x, 1)
        gamma, beta = torch.chunk(self.global_affine(stats), 2, dim=1)
        return local_delta + gamma * ms_up + beta


class SDEMLite(nn.Module):
    """Lightweight spatial detail enhancement branch.

    PAN feature details are decomposed/recomposed in wavelet space and added as
    a zero-initialized residual before channel reduction. It approximates the
    role of WFANet's SDEM without introducing quadratic attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.band_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
                    nn.PReLU(channels),
                    nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
                    nn.Sigmoid(),
                )
                for _ in range(4)
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.proj[-1].weight)
        if self.proj[-1].bias is not None:
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, pan_feat):
        coeffs = self.dwt(pan_feat)
        c = pan_feat.shape[1]
        bands = [
            coeffs[:, :c],
            coeffs[:, c: 2 * c],
            coeffs[:, 2 * c: 3 * c],
            coeffs[:, 3 * c:],
        ]
        enhanced = [band * self.band_gates[i](band) for i, band in enumerate(bands)]
        detail = self.idwt(torch.cat(enhanced, dim=1))
        return self.proj(detail)


class ZResidualHead(nn.Module):
    """Predict wavelet subband residuals from [h, z] state at each level.

    Outputs 4 * ms_channels (LL, LH, HL, HH) as zero-initialized residuals.
    Fusion: fused_XX = old_fused_XX + beta_XX * pred_r_XX
    """
    def __init__(self, state_dim, latent_dim, ms_channels):
        super().__init__()
        in_dim = state_dim + latent_dim
        out_dim = ms_channels * 4
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True),
            nn.PReLU(in_dim),
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True),
            nn.PReLU(in_dim),
            nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.beta_ll = nn.Parameter(torch.zeros(1))
        self.beta_hf = nn.Parameter(torch.zeros(1))

    def forward(self, h_state, z_state):
        x = torch.cat([h_state, z_state], dim=1)
        r = self.body(x)
        c = r.shape[1] // 4
        r_ll = r[:, :c]
        r_lh = r[:, c:2*c]
        r_hl = r[:, 2*c:3*c]
        r_hh = r[:, 3*c:]
        return r_ll, r_lh, r_hl, r_hh


class LocalFrequencyMixer(nn.Module):
    """Low-complexity local residual mixer for one high-frequency subband.

    The module starts as an exact no-op because the last projection is
    zero-initialized, but it can learn a local correction from observable
    frequency cues. Complexity is O(HW * C * k^2), not global attention.
    """
    def __init__(self, channels, kernel_size=3, hidden_scale=1.0):
        super().__init__()
        hidden = max(channels, int(round(channels * float(hidden_scale))))
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels * 6, hidden, 1, 1, 0, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, kernel_size, 1, padding, groups=hidden, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, ms_hf, pan_hf, alpha, fused_ll, ms_ll):
        alpha_pan = alpha * pan_hf
        diff = (ms_hf - pan_hf).abs()
        x = torch.cat([ms_hf, pan_hf, alpha_pan, diff, fused_ll, ms_ll], dim=1)
        return self.scale * self.body(x)


class ChannelHaarSpectralAdapter(nn.Module):
    """1D channel-wise Haar adapter for the MS residual stream.

    The report argues that MS bands should keep explicit spectral structure.
    This module exposes pairwise low/high spectral responses while starting as
    an exact no-op, so old checkpoints remain behaviorally unchanged at load.
    """
    def __init__(self, channels, hidden_channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, hidden_channels, 1, 1, 0, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def channel_haar(x):
        c = x.shape[1]
        if c < 2:
            return x
        if c % 2 != 0:
            # The known WV3/GF2/QB configs are even-channel, but keep a safe
            # fallback so the module is not brittle for custom data.
            x_pair = x[:, :-1]
            tail = x[:, -1:]
        else:
            x_pair = x
            tail = None
        even = x_pair[:, 0::2]
        odd = x_pair[:, 1::2]
        scale = 2.0 ** -0.5
        low = (even + odd) * scale
        high = (even - odd) * scale
        out = torch.cat([low, high], dim=1)
        return torch.cat([out, tail], dim=1) if tail is not None else out

    def forward(self, ms_up, lms, pan):
        spectral = self.channel_haar(ms_up)
        delta = self.body(torch.cat([ms_up, lms, spectral, pan], dim=1))
        return ms_up + self.scale * delta


class WindowedFrequencyMixer(nn.Module):
    """Windowed selective-scan mixer for local frequency correction.

    This is a dependency-free, linear-complexity proxy for the report's
    WSLM/FMamba idea: it scans short local windows in frequency subbands and
    predicts a zero-initialized residual correction.
    """
    def __init__(self, channels, window_size=8, hidden_scale=1.0):
        super().__init__()
        hidden = max(channels, int(round(channels * float(hidden_scale))))
        self.window_size = int(window_size)
        self.in_proj = nn.Conv2d(channels * 6, hidden * 3, 1, 1, 0, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=True)
        self.out_proj = nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        self.scale = nn.Parameter(torch.ones(1))

    def _window_flatten(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (-h) % ws
        pad_w = (-w) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b * (hp // ws) * (wp // ws), c, ws * ws)
        return x, (b, c, h, w, hp, wp)

    def _window_unflatten(self, x, meta):
        b, c, h, w, hp, wp = meta
        ws = self.window_size
        x = x.view(b, hp // ws, wp // ws, c, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def _bidirectional_scan(self, value, decay):
        v, meta = self._window_flatten(value)
        d, _ = self._window_flatten(decay)
        d = torch.sigmoid(d).clamp(0.02, 0.98)

        state = torch.zeros(v.shape[0], v.shape[1], device=v.device, dtype=v.dtype)
        forward = []
        for idx in range(v.shape[-1]):
            a = d[:, :, idx]
            state = a * state + (1.0 - a) * v[:, :, idx]
            forward.append(state)
        forward = torch.stack(forward, dim=-1)

        state = torch.zeros_like(state)
        backward = []
        for idx in range(v.shape[-1] - 1, -1, -1):
            a = d[:, :, idx]
            state = a * state + (1.0 - a) * v[:, :, idx]
            backward.append(state)
        backward = torch.stack(backward[::-1], dim=-1)
        return self._window_unflatten(0.5 * (forward + backward), meta)

    def forward(self, ms_hf, pan_hf, alpha, fused_ll, ms_ll):
        alpha_pan = alpha * pan_hf
        diff = (ms_hf - pan_hf).abs()
        x = torch.cat([ms_hf, pan_hf, alpha_pan, diff, fused_ll, ms_ll], dim=1)
        value, gate, decay = torch.chunk(self.in_proj(x), 3, dim=1)
        value = self.dw(value)
        mixed = self._bidirectional_scan(value, decay)
        return self.scale * self.out_proj(mixed * torch.sigmoid(gate))


class MambaFrequencyMixer(nn.Module):
    """True Mamba-based window mixer for high-frequency residual correction.

    The module mirrors WindowedFrequencyMixer's input/output contract but uses
    mamba_ssm's selective scan inside each local window. It is intentionally
    zero-initialized at the output projection, so enabling it starts as a no-op
    residual path and preserves old checkpoints as much as possible.
    """
    def __init__(
        self,
        channels,
        window_size=8,
        hidden_scale=1.0,
        d_state=16,
        d_conv=4,
        expand=2,
        bidirectional=True,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for --use-mamba-frequency-mixer. "
                "Use the wfanet_mamba environment or disable this flag."
            )
        hidden = max(channels, int(round(channels * float(hidden_scale))))
        self.window_size = int(window_size)
        self.bidirectional = bool(bidirectional)
        self.in_proj = nn.Sequential(
            nn.Conv2d(channels * 6, hidden, 1, 1, 0, bias=True),
            nn.PReLU(hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.mamba = Mamba(
            d_model=hidden,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.out_proj = nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        self.scale = nn.Parameter(torch.ones(1))

    def _window_flatten(self, x):
        b, c, h, w = x.shape
        ws = max(1, self.window_size)
        pad_h = (-h) % ws
        pad_w = (-w) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b * (hp // ws) * (wp // ws), ws * ws, c)
        return x, (b, c, h, w, hp, wp)

    def _window_unflatten(self, x, meta):
        b, c, h, w, hp, wp = meta
        ws = max(1, self.window_size)
        x = x.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, ms_hf, pan_hf, alpha, fused_ll, ms_ll):
        alpha_pan = alpha * pan_hf
        diff = (ms_hf - pan_hf).abs()
        x = torch.cat([ms_hf, pan_hf, alpha_pan, diff, fused_ll, ms_ll], dim=1)
        x = self.in_proj(x)
        seq, meta = self._window_flatten(x)
        seq = self.norm(seq)
        y = self.mamba(seq)
        if self.bidirectional:
            y_rev = torch.flip(self.mamba(torch.flip(seq, dims=[1])), dims=[1])
            y = 0.5 * (y + y_rev)
        mixed = self._window_unflatten(y, meta)
        return self.scale * self.out_proj(mixed)


class RSSMWaveletFusionHz(nn.Module):
    def __init__(self, pan_channels_per_level, ms_channels_per_level, hidden_dim=96, latent_dim=32, levels=3,
                 deterministic_only=False, separate_subband_gates=True, use_conv_gru=False,
                 learnable_fusion=False, signed_hf_gate=False, hf_gate_scale=1.0,
                 residual_learnable_fusion=False, use_state_spatial_mixer=False,
                 use_level_ll_corr=False, z_eval_mode="prior", z_update_order="legacy",
                 z_zero_levels=None, use_z_residual_head=False,
                 use_local_freq_mixer=False, lfm_kernel_size=3, lfm_hidden_scale=1.0,
                 use_windowed_freq_mixer=False, wfm_window_size=8, wfm_hidden_scale=1.0,
                 use_mamba_freq_mixer=False, mamba_window_size=8, mamba_hidden_scale=1.0,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.deterministic_only = deterministic_only
        self.use_conv_gru = use_conv_gru
        self.learnable_fusion = learnable_fusion
        self.signed_hf_gate = signed_hf_gate
        self.hf_gate_scale = hf_gate_scale
        self.residual_learnable_fusion = residual_learnable_fusion
        self.use_state_spatial_mixer = use_state_spatial_mixer
        self.use_level_ll_corr = use_level_ll_corr
        self.z_eval_mode = z_eval_mode
        self.z_update_order = z_update_order
        self.z_zero_levels = set(z_zero_levels or [])
        self.collect_z_diagnostics = False
        self.last_z_diagnostics = []
        self.use_z_residual_head = use_z_residual_head
        self.use_local_freq_mixer = use_local_freq_mixer
        self.use_windowed_freq_mixer = use_windowed_freq_mixer
        self.use_mamba_freq_mixer = use_mamba_freq_mixer

        if use_z_residual_head:
            self.z_res_heads = nn.ModuleList([
                ZResidualHead(hidden_dim, latent_dim, ms_channels_per_level[i])
                for i in range(levels)
            ])

        if use_local_freq_mixer:
            self.local_mixer_lh = nn.ModuleList([
                LocalFrequencyMixer(ms_channels_per_level[i], lfm_kernel_size, lfm_hidden_scale)
                for i in range(levels)
            ])
            self.local_mixer_hl = nn.ModuleList([
                LocalFrequencyMixer(ms_channels_per_level[i], lfm_kernel_size, lfm_hidden_scale)
                for i in range(levels)
            ])
            self.local_mixer_hh = nn.ModuleList([
                LocalFrequencyMixer(ms_channels_per_level[i], lfm_kernel_size, lfm_hidden_scale)
                for i in range(levels)
            ])

        if use_windowed_freq_mixer:
            self.window_mixer_lh = nn.ModuleList([
                WindowedFrequencyMixer(ms_channels_per_level[i], wfm_window_size, wfm_hidden_scale)
                for i in range(levels)
            ])
            self.window_mixer_hl = nn.ModuleList([
                WindowedFrequencyMixer(ms_channels_per_level[i], wfm_window_size, wfm_hidden_scale)
                for i in range(levels)
            ])

        if use_mamba_freq_mixer:
            self.mamba_mixer_lh = nn.ModuleList([
                MambaFrequencyMixer(
                    ms_channels_per_level[i], mamba_window_size, mamba_hidden_scale,
                    mamba_d_state, mamba_d_conv, mamba_expand
                )
                for i in range(levels)
            ])
            self.mamba_mixer_hl = nn.ModuleList([
                MambaFrequencyMixer(
                    ms_channels_per_level[i], mamba_window_size, mamba_hidden_scale,
                    mamba_d_state, mamba_d_conv, mamba_expand
                )
                for i in range(levels)
            ])
            self.mamba_mixer_hh = nn.ModuleList([
                MambaFrequencyMixer(
                    ms_channels_per_level[i], mamba_window_size, mamba_hidden_scale,
                    mamba_d_state, mamba_d_conv, mamba_expand
                )
                for i in range(levels)
            ])
            self.window_mixer_hh = nn.ModuleList([
                WindowedFrequencyMixer(ms_channels_per_level[i], wfm_window_size, wfm_hidden_scale)
                for i in range(levels)
            ])

        self.fusion_blocks = nn.ModuleList(
            [
                CrossScaleFusionHz(
                    pan_channels=pan_channels_per_level[i],
                    ms_channels=ms_channels_per_level[i],
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    deterministic_only=deterministic_only,
                    use_conv_gru=use_conv_gru,
                    use_state_spatial_mixer=use_state_spatial_mixer,
                    use_level_ll_corr=use_level_ll_corr,
                    z_eval_mode=z_eval_mode,
                    z_update_order=z_update_order,
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
            if residual_learnable_fusion:
                self.conv_fusion_beta_lh = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])
                self.conv_fusion_beta_hl = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])
                self.conv_fusion_beta_hh = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(levels)])

        self.wavelet_reconstructor = WaveletReconstructor(levels=levels)
        self.refine = nn.Sequential(resblock(ms_channels_per_level[0]), resblock(ms_channels_per_level[0]))

    def _hf_alpha(self, alpha):
        if self.signed_hf_gate:
            # Keep the pretrained positive gate and add a small signed correction.
            # This can suppress or amplify PAN details without destroying old checkpoints at step 0.
            return alpha + self.hf_gate_scale * (2.0 * alpha - 1.0)
        return self.hf_gate_scale * alpha

    def _fuse_high(self, ms_hf, pan_hf, alpha, conv_fusion=None, beta=None):
        if self.learnable_fusion:
            delta = conv_fusion(torch.cat([ms_hf, pan_hf, alpha * pan_hf], dim=1))
            if self.residual_learnable_fusion:
                return ms_hf + alpha * pan_hf + beta * delta
            return delta
        return ms_hf + alpha * pan_hf

    def set_z_eval_mode(self, mode):
        if mode not in {"prior", "posterior", "zero"}:
            raise ValueError(f"Unsupported z_eval_mode: {mode}")
        self.z_eval_mode = mode
        for block in self.fusion_blocks:
            block.set_z_eval_mode(mode)

    def set_z_diagnostics(self, enabled=True):
        self.collect_z_diagnostics = bool(enabled)

    def get_z_diagnostics(self):
        return list(self.last_z_diagnostics)

    def forward(self, pan_pyramid, ms_pyramid, training=True):
        h_state = None
        z_state = None
        fused_coeffs = []
        kl_terms = []
        z_residuals = [] if self.use_z_residual_head else None
        self.last_z_diagnostics = []

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

            force_zero_z = level in self.z_zero_levels
            fused_ll, h_state, z_state, kl_mean, fused_ll_z0 = self.fusion_blocks[level](
                pan_feat,
                ms_feat,
                h_state,
                z_state,
                training=training,
                force_zero_z=force_zero_z,
                return_z0=self.collect_z_diagnostics,
            )
            kl_terms.append(kl_mean)

            pan_lh = self.pan_high_to_ms[level]["lh"](lh_pan)
            pan_hl = self.pan_high_to_ms[level]["hl"](hl_pan)
            pan_hh = self.pan_high_to_ms[level]["hh"](hh_pan)

            z_gate = self.z_to_gate[level](z_state)
            gate_in = torch.cat([fused_ll, ll_ms, pan_lh, pan_hl, pan_hh, z_gate], dim=1)
            z_gate_zero = None
            gate_in_z0 = None
            if self.collect_z_diagnostics:
                z_gate_zero = self.z_to_gate[level](torch.zeros_like(z_state))
                fused_ll_base = fused_ll_z0 if fused_ll_z0 is not None else fused_ll
                gate_in_z0 = torch.cat([fused_ll_base, ll_ms, pan_lh, pan_hl, pan_hh, z_gate_zero], dim=1)

            if self.separate_subband_gates:
                alpha_lh = self._hf_alpha(self.high_gate_lh[level](gate_in))
                alpha_hl = self._hf_alpha(self.high_gate_hl[level](gate_in))
                alpha_hh = self._hf_alpha(self.high_gate_hh[level](gate_in))
                if self.collect_z_diagnostics:
                    alpha_lh_z0 = self._hf_alpha(self.high_gate_lh[level](gate_in_z0))
                    alpha_hl_z0 = self._hf_alpha(self.high_gate_hl[level](gate_in_z0))
                    alpha_hh_z0 = self._hf_alpha(self.high_gate_hh[level](gate_in_z0))
                beta_lh = self.conv_fusion_beta_lh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hl = self.conv_fusion_beta_hl[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hh = self.conv_fusion_beta_hh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                fused_lh = self._fuse_high(
                    lh_ms, pan_lh, alpha_lh, self.conv_fusion_lh[level] if self.learnable_fusion else None, beta_lh
                )
                fused_hl = self._fuse_high(
                    hl_ms, pan_hl, alpha_hl, self.conv_fusion_hl[level] if self.learnable_fusion else None, beta_hl
                )
                fused_hh = self._fuse_high(
                    hh_ms, pan_hh, alpha_hh, self.conv_fusion_hh[level] if self.learnable_fusion else None, beta_hh
                )
            else:
                alpha = self._hf_alpha(self.high_gate[level](gate_in))
                if self.collect_z_diagnostics:
                    alpha_z0 = self._hf_alpha(self.high_gate[level](gate_in_z0))
                    alpha_lh_z0 = alpha_z0
                    alpha_hl_z0 = alpha_z0
                    alpha_hh_z0 = alpha_z0
                    alpha_lh = alpha
                    alpha_hl = alpha
                    alpha_hh = alpha
                beta_lh = self.conv_fusion_beta_lh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hl = self.conv_fusion_beta_hl[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                beta_hh = self.conv_fusion_beta_hh[level] if self.learnable_fusion and self.residual_learnable_fusion else None
                fused_lh = self._fuse_high(
                    lh_ms, pan_lh, alpha, self.conv_fusion_lh[level] if self.learnable_fusion else None, beta_lh
                )
                fused_hl = self._fuse_high(
                    hl_ms, pan_hl, alpha, self.conv_fusion_hl[level] if self.learnable_fusion else None, beta_hl
                )
                fused_hh = self._fuse_high(
                    hh_ms, pan_hh, alpha, self.conv_fusion_hh[level] if self.learnable_fusion else None, beta_hh
                )

            if self.use_local_freq_mixer:
                fused_lh = fused_lh + self.local_mixer_lh[level](lh_ms, pan_lh, alpha_lh, fused_ll, ll_ms)
                fused_hl = fused_hl + self.local_mixer_hl[level](hl_ms, pan_hl, alpha_hl, fused_ll, ll_ms)
                fused_hh = fused_hh + self.local_mixer_hh[level](hh_ms, pan_hh, alpha_hh, fused_ll, ll_ms)
            if self.use_windowed_freq_mixer:
                fused_lh = fused_lh + self.window_mixer_lh[level](lh_ms, pan_lh, alpha_lh, fused_ll, ll_ms)
                fused_hl = fused_hl + self.window_mixer_hl[level](hl_ms, pan_hl, alpha_hl, fused_ll, ll_ms)
                fused_hh = fused_hh + self.window_mixer_hh[level](hh_ms, pan_hh, alpha_hh, fused_ll, ll_ms)
            if self.use_mamba_freq_mixer:
                fused_lh = fused_lh + self.mamba_mixer_lh[level](lh_ms, pan_lh, alpha_lh, fused_ll, ll_ms)
                fused_hl = fused_hl + self.mamba_mixer_hl[level](hl_ms, pan_hl, alpha_hl, fused_ll, ll_ms)
                fused_hh = fused_hh + self.mamba_mixer_hh[level](hh_ms, pan_hh, alpha_hh, fused_ll, ll_ms)

            # Z residual auxiliary head: apply zero-init residual correction
            z_res_pred = None
            if self.use_z_residual_head:
                r_ll, r_lh, r_hl, r_hh = self.z_res_heads[level](h_state, z_state)
                z_res_pred = (r_ll, r_lh, r_hl, r_hh)
                z_residuals.append(z_res_pred)
                fused_ll = fused_ll + self.z_res_heads[level].beta_ll * r_ll
                fused_lh = fused_lh + self.z_res_heads[level].beta_hf * r_lh
                fused_hl = fused_hl + self.z_res_heads[level].beta_hf * r_hl
                fused_hh = fused_hh + self.z_res_heads[level].beta_hf * r_hh

            fused_coeffs.append((fused_ll, fused_lh, fused_hl, fused_hh))

            if self.collect_z_diagnostics:
                ll_delta = (
                    (fused_ll - fused_ll_z0).abs().mean()
                    if fused_ll_z0 is not None
                    else fused_ll.new_tensor(0.0)
                )
                self.last_z_diagnostics.append(
                    {
                        "level": int(level),
                        "force_zero_z": bool(force_zero_z),
                        "z_abs_mean": float(z_state.detach().abs().mean().cpu()),
                        "z_std": float(z_state.detach().float().std(unbiased=False).cpu()),
                        "z_gate_abs_mean": float(z_gate.detach().abs().mean().cpu()),
                        "ll_delta_z0": float(ll_delta.detach().cpu()),
                        "alpha_lh_mean": float(alpha_lh.detach().mean().cpu()),
                        "alpha_hl_mean": float(alpha_hl.detach().mean().cpu()),
                        "alpha_hh_mean": float(alpha_hh.detach().mean().cpu()),
                        "alpha_lh_delta_z0": float((alpha_lh - alpha_lh_z0).detach().abs().mean().cpu()),
                        "alpha_hl_delta_z0": float((alpha_hl - alpha_hl_z0).detach().abs().mean().cpu()),
                        "alpha_hh_delta_z0": float((alpha_hh - alpha_hh_z0).detach().abs().mean().cpu()),
                        "kl_mean": float(kl_mean.detach().cpu()),
                    }
                )

        fused_coeffs = list(reversed(fused_coeffs))
        recon = self.wavelet_reconstructor(fused_coeffs)
        recon = self.refine(recon)

        kl_loss = torch.stack(kl_terms).mean() if kl_terms else recon.new_tensor(0.0)
        return recon, kl_loss, z_residuals


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
        use_lowfreq_corr=False,
        signed_hf_gate=False,
        hf_gate_scale=1.0,
        residual_learnable_fusion=False,
        use_sdem_lite=False,
        use_state_spatial_mixer=False,
        use_level_ll_corr=False,
        use_band_corr=False,
        band_corr_kernel_size=5,
        band_corr_hidden=32,
        z_eval_mode="prior",
        z_update_order="legacy",
        z_zero_levels=None,
        levels=3,
        use_z_residual_head=False,
        use_local_freq_mixer=False,
        lfm_kernel_size=3,
        lfm_hidden_scale=1.0,
        use_windowed_freq_mixer=False,
        wfm_window_size=8,
        wfm_hidden_scale=1.0,
        use_mamba_freq_mixer=False,
        mamba_window_size=8,
        mamba_hidden_scale=1.0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        use_channel_dwt_adapter=False,
        channel_dwt_hidden=32,
    ):
        super().__init__()
        self.deterministic_only = deterministic_only
        self.image_space_wavelet = image_space_wavelet
        self.levels = levels
        self.use_sdem_lite = use_sdem_lite
        self.z_eval_mode = z_eval_mode
        self.z_update_order = z_update_order
        self.use_channel_dwt_adapter = use_channel_dwt_adapter

        self.pan_raise = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
        self.ms_upsample = nn.Sequential(
            nn.Conv2d(L_up_channel, L_up_channel * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        self.ms_act = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.channel_dwt_adapter = (
            ChannelHaarSpectralAdapter(L_up_channel, hidden_channels=channel_dwt_hidden)
            if use_channel_dwt_adapter else None
        )
        self.ms_raise = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)

        self.wavelet = WaveletPyramid(levels=levels)

        if image_space_wavelet:
            # Wavelet on raw images: PAN(1-ch) and MS_up(L_up_channel-ch).
            # Each PAN subband (1-ch) is raised to pan_target_channel via a shared raiser.
            self.pan_subband_raise = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
            # Each MS subband (L_up_channel-ch) is raised to ms_target_channel via a shared raiser.
            self.ms_subband_raise = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)

        pan_channels_per_level = [pan_target_channel * 4] * levels
        ms_channels_per_level = [ms_target_channel] * levels

        self.rssm_fusion = RSSMWaveletFusionHz(
            pan_channels_per_level=pan_channels_per_level,
            ms_channels_per_level=ms_channels_per_level,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            levels=levels,
            deterministic_only=deterministic_only,
            separate_subband_gates=separate_subband_gates,
            use_conv_gru=use_conv_gru,
            learnable_fusion=learnable_fusion,
            signed_hf_gate=signed_hf_gate,
            hf_gate_scale=hf_gate_scale,
            residual_learnable_fusion=residual_learnable_fusion,
            use_state_spatial_mixer=use_state_spatial_mixer,
            use_level_ll_corr=use_level_ll_corr,
            z_eval_mode=z_eval_mode,
            z_update_order=z_update_order,
            z_zero_levels=z_zero_levels,
            use_z_residual_head=use_z_residual_head,
            use_local_freq_mixer=use_local_freq_mixer,
            lfm_kernel_size=lfm_kernel_size,
            lfm_hidden_scale=lfm_hidden_scale,
            use_windowed_freq_mixer=use_windowed_freq_mixer,
            wfm_window_size=wfm_window_size,
            wfm_hidden_scale=wfm_hidden_scale,
            use_mamba_freq_mixer=use_mamba_freq_mixer,
            mamba_window_size=mamba_window_size,
            mamba_hidden_scale=mamba_hidden_scale,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )

        self.reduce = reduce_channel(ms_target_channel=ms_target_channel, L_up_channel=L_up_channel)
        self.lowfreq_corr = LowFreqCorrection(L_up_channel) if use_lowfreq_corr else None
        self.band_corr = BandAwareCorrection(
            L_up_channel,
            hidden_channels=band_corr_hidden,
            kernel_size=band_corr_kernel_size,
        ) if use_band_corr else None
        self.sdem_lite = SDEMLite(ms_target_channel) if use_sdem_lite else None
        self.out_act = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.fused_weight = nn.Parameter(torch.ones(1))  # default 1.0 = original behavior

    def set_z_eval_mode(self, mode):
        self.z_eval_mode = mode
        self.rssm_fusion.set_z_eval_mode(mode)

    def set_z_diagnostics(self, enabled=True):
        self.rssm_fusion.set_z_diagnostics(enabled)

    def get_z_diagnostics(self):
        return self.rssm_fusion.get_z_diagnostics()

    def forward(self, pan, ms, lms):
        ms_up = self.ms_upsample(ms)
        ms_up = self.ms_act(ms_up + lms)
        if self.channel_dwt_adapter is not None:
            ms_up = self.channel_dwt_adapter(ms_up, lms, pan)

        if self.image_space_wavelet:
            pan_feat_for_sdem = self.pan_raise(pan) if self.sdem_lite is not None else None
            # Wavelet decompose raw PAN and MS_up in image space.
            pan_pyr_raw = self.wavelet(pan)
            ms_pyr_raw = self.wavelet(ms_up)

            # Raise each subband from image-space channels to feature-space channels.
            pan_pyr = []
            ms_pyr = []
            for level in range(self.levels):
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
            pan_feat_for_sdem = pan_feat
            pan_pyr = self.wavelet(pan_feat)
            ms_pyr = self.wavelet(ms_feat)

        fused, kl_loss, z_residuals = self.rssm_fusion(pan_pyr, ms_pyr, training=self.training)
        if self.sdem_lite is not None:
            fused = fused + self.sdem_lite(pan_feat_for_sdem)
        fused = self.reduce(fused)
        delta_lf = self.lowfreq_corr(fused, ms_up, pan) if self.lowfreq_corr is not None else 0.0
        base = self.fused_weight * fused + ms_up + delta_lf
        delta_band = self.band_corr(base, fused, ms_up, pan) if self.band_corr is not None else 0.0
        out = self.out_act(base + delta_band)
        return out, kl_loss, z_residuals


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
    out, kl, _ = model(pan, ms, lms)
    print(out.shape, kl.item())
