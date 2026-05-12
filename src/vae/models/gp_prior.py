import math
from typing import Tuple, Optional

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, UnwhitenedVariationalStrategy
from gpytorch.kernels import RQKernel, LinearKernel, ProductKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions import MultivariateNormal

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rq_kernel(
    active_dims: Tuple[int, ...],
    batch_shape: torch.Size,
    lengthscale_loc: float = math.log(0.2),  # inputs normalized to [0,1]
    lengthscale_scale: float = 0.5,
) -> RQKernel:
    """Rational Quadratic with ARD and reasonable priors/constraints."""
    return RQKernel(
        active_dims=active_dims,
        batch_shape=batch_shape,
        lengthscale_prior=gpytorch.priors.LogNormalPrior(lengthscale_loc, lengthscale_scale),
        lengthscale_constraint=gpytorch.constraints.GreaterThan(0.03),
        alpha_prior=gpytorch.priors.GammaPrior(2.0, 1.0),
        alpha_constraint=gpytorch.constraints.GreaterThan(0.05),
    )


def create_inducing_points(
    cycle_range: Tuple[float, float],
    rate_range:  Tuple[float, float],
    num_inducing: int = 64,
    strategy: str = "sobol",   # "sobol" | "sobol_biased" | "adaptive_grid" | "random"
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Return [M, 2] inducing points in (cycle, rate) on target device.
    """
    c0, c1 = cycle_range
    r0, r1 = rate_range

    if strategy == "sobol":
        sob = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        pts = sob.draw(num_inducing).to(device=device, dtype=torch.float32)
        c = c0 + pts[:, 0] * (c1 - c0)
        r = r0 + pts[:, 1] * (r1 - r0)
        return torch.stack([c, r], dim=-1)

    if strategy == "sobol_biased":
        # Half global, half concentrated in a mid-rate band (example [0.45, 0.75])
        m1 = num_inducing // 2
        m2 = num_inducing - m1

        sob1 = torch.quasirandom.SobolEngine(2, scramble=True)
        p1 = sob1.draw(m1).to(device=device, dtype=torch.float32)
        c1s = c0 + p1[:, 0] * (c1 - c0)
        r1s = r0 + p1[:, 1] * (r1 - r0)

        sob2 = torch.quasirandom.SobolEngine(2, scramble=True)
        p2 = sob2.draw(m2).to(device=device, dtype=torch.float32)
        c2s = c0 + p2[:, 0] * (c1 - c0)
        r_low, r_high = 0.45, 0.75
        r2s = r_low + p2[:, 1] * (r_high - r_low)

        return torch.stack([torch.cat([c1s, c2s]), torch.cat([r1s, r2s])], dim=-1)

    if strategy == "adaptive_grid":
        side = max(int(num_inducing ** 0.5), 2)
        c_vals = torch.linspace(c0, c1, side, device=device)
        r_vals = torch.linspace(r0, r1, side, device=device)
        grid_c, grid_r = torch.meshgrid(c_vals, r_vals, indexing="ij")
        pts = torch.stack([grid_c.reshape(-1), grid_r.reshape(-1)], dim=-1)
        return pts[:num_inducing]

    if strategy == "random":
        c = torch.rand(num_inducing, device=device) * (c1 - c0) + c0
        r = torch.rand(num_inducing, device=device) * (r1 - r0) + r0
        return torch.stack([c, r], dim=-1)

    raise ValueError(f"Unknown inducing strategy: {strategy}")


@torch.no_grad()
def log_gp_hypers(
    gp_model: "BatchedGPModel",
    prefix: str = "[gp]",
    logger: Optional[object] = None,
) -> None:
    """
    Log typical GP hyperparameters for monitoring convergence.

    Logs (per epoch, when called):
      - component kernels under the ScaleKernel
      - mean outputscale over tasks
      - mean RQ lengthscales across tasks for (cycle, rate)

    Uses the provided `logger` if given, otherwise falls back to print.
    """
    def _log(msg: str) -> None:
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    try:
        base = gp_model.covar_module.base_kernel  # sum/product of kernels
        comps = getattr(base, "kernels", [])
        _log(f"{prefix} base components: {[k.__class__.__name__ for k in comps]}")

        outscale = gp_model.covar_module.outputscale.detach().cpu()
        _log(
            f"{prefix} outputscale mean={outscale.mean().item():.4f} "
            f"min={outscale.min().item():.4f} max={outscale.max().item():.4f}"
        )

        # RQ per-task per-dim lengthscales if present
        for idx, k in enumerate(comps):
            if isinstance(k, RQKernel):
                ls = k.lengthscale.detach().cpu().squeeze(-2)  # [D, 2]
                if ls.ndim == 1:
                    ls = ls.unsqueeze(0)
                m = ls.mean(0).tolist()
                _log(
                    f"{prefix} RQ[{idx}] mean lengthscale "
                    f"(cycle, rate) = ({m[0]:.4f}, {m[1]:.4f})"
                )
    except Exception:
        # keep training robust even if logging fails
        return

# ---------------------------------------------------------------------
# Model (Pattern 1): Single VS with batch_shape=[D]; input [B,2] -> output.mean is [B, D]
# ---------------------------------------------------------------------

class BatchedGPModel(ApproximateGP):
    """
    Pattern 1:
      - Per-task GP via batch_shape=[D] (no MultitaskVariationalStrategy).
      - One set of inputs X=[B,2] shared across D tasks.
      - Returns a MultitaskMultivariateNormal whose .mean/.variance are [B, D].
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,   # [M, 2]
        num_latent_dims: int,
        include_linear: bool = True,
        jitter: float = 1e-4,
    ) -> None:
        assert inducing_points.ndim == 2 and inducing_points.size(1) == 2, \
            f"inducing_points must be [M,2], got {tuple(inducing_points.shape)}"
        M = int(inducing_points.size(0))
        D = int(num_latent_dims)
        assert M > 0, "No inducing points (M==0)."

        ip = inducing_points

        # One q(u) per task (batched)
        q = CholeskyVariationalDistribution(M, batch_shape=torch.Size([D]), jitter=jitter)

        vs = UnwhitenedVariationalStrategy(
            self,
            ip, 
            q,
            learn_inducing_locations=True,
        )
        super().__init__(vs)
        
        base_vs = self.variational_strategy  # plain VS here
        ip_param = getattr(base_vs, "inducing_points", None)
        qdist    = getattr(base_vs, "_variational_distribution", None)
        
        assert ip_param is not None, "VS has no inducing_points"
        assert qdist    is not None, "VS has no variational distribution"
        
        vm = qdist.variational_mean  # [D, M]
        M_ip = ip_param.size(-2)
        M_q  = vm.size(-1)
        
        print(f"[GP sanity] ip={tuple(ip_param.shape)}  q.mean={tuple(vm.shape)}  (expect ip[* , M, 2], q[D, M])")
        assert M_ip > 0, f"Inducing points have zero rows: {tuple(ip_param.shape)}"
        assert M_q  > 0, f"q(u) has zero event size: {tuple(vm.shape)}"
        assert M_q == M_ip, f"M mismatch: ip has M={M_ip}, q has M={M_q}"

        self.num_latent_dims = D
        self.jitter = float(jitter)

        # Kernels: RQ(cycle) + RQ(rate) + Product(RQ(cycle), RQ(rate)) + optional Linear
        bshape = torch.Size([D])
        CYCLE, RATE = 0, 1

        k_cycle = _rq_kernel((CYCLE,), bshape)
        k_rate  = _rq_kernel((RATE,),  bshape)
        k_inter = ProductKernel(_rq_kernel((CYCLE,), bshape), _rq_kernel((RATE,), bshape))

        base = k_cycle + k_rate + k_inter
        if include_linear:
            base = base + LinearKernel(ard_num_dims=2, batch_shape=bshape)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=bshape)
        self.covar_module = ScaleKernel(
            base,
            batch_shape=bshape,
            outputscale_prior=gpytorch.priors.LogNormalPrior(0.0, 0.5),  # ~1.0
            outputscale_constraint=gpytorch.constraints.GreaterThan(1e-6),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        x: [B, 2] (cycle_norm, charge_rate)
        Return a *batched* MVN with batch shape [D] and event size B.
        This makes prior.mean at inducing inputs shape [D, M],
        which matches q(u).variational_mean: [D, M].
        """
        D = self.num_latent_dims
        assert x.ndim == 2 and x.size(-1) == 2, f"x must be [B,2], got {tuple(x.shape)}"
        B = x.size(0)
    
        # Mean: make sure final is [D, B]
        mean_x = self.mean_module(x)  # can be [B], [D,B], or [B,D] depending on broadcast
        if mean_x.ndim == 1:                       # [B] -> [D,B]
            mean_x = mean_x.unsqueeze(0).expand(D, B)
        elif mean_x.shape == (B, D):               # [B,D] -> [D,B]
            mean_x = mean_x.transpose(0, 1).contiguous()
        elif mean_x.shape == (D, B):               # already [D,B]
            pass
        else:
            raise RuntimeError(f"Unexpected mean shape {tuple(mean_x.shape)} for x={tuple(x.shape)}")
    
        # Covariance: batch [D], event [B,B]
        covar_x = self.covar_module(x).add_jitter(self.jitter)
    
        # IMPORTANT: return a batched MVN, not MultitaskMultivariateNormal
        return MultivariateNormal(mean_x, covar_x)
# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_gp_prior_model(
    config,
    device: torch.device,
    cycle_range: Tuple[float, float] = (0.0, 1.0),
    rate_range:  Tuple[float, float] = (0.0, 1.0),
    inducing_strategy: Optional[str] = None,   # "sobol" | "sobol_biased" | "adaptive_grid" | "random"
):
    """
    Construct the batched GP prior, likelihood, and ELBO.
    - Input x is [B, 2].
    - Output gp_posterior.mean / .variance are [B, D] (no transpose needed).
    """
    hp = config.HYPER_PARAMETERS
    gp_cfg = getattr(hp, "gp_schedule", None)

    D = int(getattr(hp, "latent_dim", 3))
    M = int(getattr(gp_cfg, "num_inducing", 64)) if gp_cfg is not None else 64
    include_linear  = bool(getattr(gp_cfg, "include_linear", False)) if gp_cfg is not None else False
    like_noise      = float(getattr(gp_cfg, "likelihood_noise", 1e-3)) if gp_cfg is not None else 1e-3
    mll_num_data    = int(getattr(gp_cfg, "num_data", 1)) if gp_cfg is not None else 1
    strategy        = inducing_strategy or (str(getattr(gp_cfg, "inducing_strategy", "sobol")) if gp_cfg is not None else "sobol")

    # [M,2] inducing points (cycle, rate)
    inducing_points = create_inducing_points(
        cycle_range=cycle_range,
        rate_range=rate_range,
        num_inducing=M,
        strategy=strategy,
        device=device,
    )

    gp_model = BatchedGPModel(
        inducing_points=inducing_points,
        num_latent_dims=D,
        include_linear=include_linear,
        jitter=1e-4,
    ).to(device)

    likelihood = GaussianLikelihood(
        num_tasks=D,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
    ).to(device)

    # Initialize per-task noises
    with torch.no_grad():
        try:
            likelihood.task_noises = torch.full((D,), like_noise, device=device)
        except Exception:
            # version fallback
            raw = getattr(likelihood, "raw_task_noises", None)
            if raw is not None:
                raw.data.copy_(torch.full_like(raw.data, math.log(like_noise)))

    # Try to register a prior on task noise (version-safe)
    try:
        likelihood.task_noise_covar.register_prior(
            "task_noise_prior", gpytorch.priors.LogNormalPrior(-7.0, 0.5), "noise"
        )
    except Exception:
        try:
            likelihood.task_noise_covar.register_prior(
                "task_noise_prior", gpytorch.priors.LogNormalPrior(-7.0, 0.5), "raw_task_noises"
            )
        except Exception:
            pass

    mll = VariationalELBO(likelihood, gp_model, num_data=mll_num_data)
    return gp_model, likelihood, mll
