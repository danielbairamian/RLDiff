import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Literal, Optional
import time

class MonteCarloLayer(nn.Module):
    r"""
    Monte Carlo Dropout Layer with optional reweighting via an Auxiliary Score network or Attention over MC samples.

    Summary
    -------
    Wraps a block `block: nn.Module` with dropout-based stochastic sampling. For each input x,
    we draw `mc_samples` stochastic perturbations via dropout. The main block maps each perturbed
    sample to predictions {y_i}.
    
    Two reweighting modes are supported:
    1. **auxiliary**: Uses an auxiliary network to score samples, producing normalized weights via softmax.
    2. **attention**: Uses transformer-style attention to produce normalized attention weights.

    Inputs
    ------
    block : nn.Module
        Wrapped network mapping x -> y. Must accept batched input.
        IMPORTANT: Block should not contain internal dropout; dropout is applied on x.
    dropout_p : float in [0,1)
        Dropout probability applied to x (pre-block).
    mc_samples : int >= 1
        Number of stochastic forward samples.
    reweight : bool
        If True, use reweighting (auxiliary or attention mode).
        If False, return unweighted (plain MC Dropout).
    attention_mode : Literal['auxiliary', 'attention']
        - 'auxiliary': Use aux_net to produce scalar scores per sample (original SNIS).
        - 'attention': Use transformer attention over MC samples.
    attend_mode : Literal['inputs', 'outputs']
        - 'inputs': Score/attend on dropped inputs x_drop
        - 'outputs': Score/attend on model outputs y
    aux_net : Optional[nn.Module]
        Auxiliary network for 'auxiliary' mode. If None and mode='auxiliary',
        defaults to copy.deepcopy(block).
    score_reduce : Literal['mean','sum','l2','logsumexp']
        How to reduce aux output to scalar (auxiliary mode only).
    temperature : float > 0
        Softmax temperature for auxiliary mode weight sharpness.
    num_heads : int >= 1
        Number of attention heads for attention mode.
    embedding_size : Optional[int]
        Embedding dimension for Q/K/V projections in attention. If None, uses output dim of block.
        Must be divisible by num_heads.

    Forward(...)
    -----------
    x : (B, ...)
    return_stats : tuple[str,...]
        Any of: "mean","std","variance","median","samples","quantiles","iqr","range","cv","confidence"
    quantiles : tuple[float,...] in [0,1]
    confidence_method : 'inverse_cv' | 'inverse_std' | 'agreement' | 'sharpness'

    Outputs
    -------
    Dict[str, Tensor] with shapes:
        - mean, std, variance, median, iqr, range, cv, confidence: (B, ...)
        - samples: (mc_samples, B, ...)
        - quantiles: (len(quantiles), B, ...)
      All reductions are **weighted** if reweight=True.
    """

    def __init__(
        self,
        block: nn.Module,
        dropout_p: float,
        mc_samples: int = 128,
        *,
        reweight: bool = True,
        attention_mode: Literal['auxiliary', 'attention'] = 'auxiliary',
        attend_mode: Literal['inputs', 'outputs'] = 'outputs',
        aux_net: Optional[nn.Module] = None,
        score_reduce: Literal['mean', 'sum', 'l2', 'logsumexp'] = 'mean',
        temperature: float = 1.0,
        num_heads: int = 4,
        embedding_size: Optional[int] = None,
        query_mode: Literal['mean', 'per_sample'] = 'mean',
    ) -> None:
        super().__init__()
        
        # main block
        self.block = block

        # validate dropout
        self.dropout_p = float(dropout_p)
        if not 0 <= self.dropout_p < 1:
            raise ValueError(f"dropout_p must be in [0,1), got {dropout_p}")

        # validate samples
        self.mc_samples = int(mc_samples)
        if self.mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")

        # Reweighting config
        self.reweight = bool(reweight)
        self.attention_mode = attention_mode
        self.attend_mode = attend_mode
        
        if self.reweight:
            if attention_mode == 'auxiliary':
                # For auxiliary mode, we need to create/validate the aux_net
                # based on what it will attend to
                if attend_mode == 'inputs':
                    # Aux net sees dropped inputs - use provided or copy of block
                    self.aux_net = aux_net if aux_net is not None else copy.deepcopy(block)
                else:  # 'outputs'
                    # Aux net sees model outputs - need compatible network
                    if aux_net is None:
                        # Create a simple scoring network for outputs
                        dev, dt = self._module_device_dtype(block)
                        dummy_input = torch.randn(1, *self._infer_input_shape(block), device=dev, dtype=dt)
                        with torch.no_grad():
                            dummy_output = block(dummy_input)
                        output_dim = dummy_output.shape[-1] if dummy_output.dim() > 1 else dummy_output.numel()
                        
                        # Create a simple MLP to score outputs
                        self.aux_net = nn.Sequential(
                            nn.Linear(output_dim, max(output_dim // 2, 1)),
                            nn.ReLU(),
                            nn.Linear(max(output_dim // 2, 1), 1)
                        )
                    else:
                        self.aux_net = aux_net
                
                self.score_reduce = score_reduce
                if temperature <= 0:
                    raise ValueError("temperature must be > 0.")
                self.temperature = float(temperature)
                
                # attention params not used
                self.cross_attn = None
                self.num_heads = None
                self.embedding_size = None
                self.query_mode = None
                
            elif attention_mode == 'attention':
                # Infer dimension for attention based on attend_mode
                dev, dt = self._module_device_dtype(block)
                
                if attend_mode == 'inputs':
                    # Infer input dimension
                    self.feature_dim = self._infer_input_dim(block)
                elif attend_mode == 'outputs':
                    # Infer output dimension from dummy forward pass
                    dummy_input = torch.randn(1, *self._infer_input_shape(block), device=dev, dtype=dt)
                    with torch.no_grad():
                        dummy_output = block(dummy_input)
                    self.feature_dim = dummy_output.shape[-1] if dummy_output.dim() > 1 else dummy_output.numel()
                else:
                    raise ValueError(f"Unknown attend_mode: {attend_mode}")
                
                # Initialize attention module
                self.num_heads = num_heads
                # Default embedding_size to a value divisible by num_heads
                if embedding_size is None:
                    # Round feature_dim to nearest multiple of num_heads
                    self.embedding_size = ((self.feature_dim + num_heads - 1) // num_heads) * num_heads
                else:
                    self.embedding_size = embedding_size
                    
                # Validate embedding_size is divisible by num_heads
                if self.embedding_size % self.num_heads != 0:
                    raise ValueError(
                        f"embedding_size ({self.embedding_size}) must be divisible by num_heads ({self.num_heads}). "
                        f"Feature dim is {self.feature_dim}. Consider setting embedding_size to a multiple of {self.num_heads}."
                    )
                
                self.query_mode = query_mode
                self.cross_attn = AttentionScorer(
                    feature_dim=self.feature_dim,
                    num_heads=self.num_heads,
                    embedding_size=self.embedding_size,
                    query_mode=self.query_mode
                )
                
                # Auxiliary params not used
                self.aux_net = None
                self.score_reduce = None
                self.temperature = None
            else:
                raise ValueError(f"Unknown attention_mode: {attention_mode}")
        else:
            self.aux_net = None
            self.cross_attn = None
            self.score_reduce = None
            self.temperature = None
            self.num_heads = None
            self.embedding_size = None
            self.query_mode = None
            self.attend_mode = None
    
    @staticmethod
    def _module_device_dtype(mod: nn.Module):
        for p in mod.parameters(recurse=True):
            return p.device, p.dtype
        for b in mod.buffers(recurse=True):
            return b.device, b.dtype
        # fallback if the block is parameterless
        return torch.device("cpu"), torch.float32

    def _infer_input_shape(self, block: nn.Module):
        """Infer the input shape for the block (simple heuristic)."""
        # Try to find first Linear layer to get input dim
        for module in block.modules():
            if isinstance(module, nn.Linear):
                return (module.in_features,)
        # Default fallback
        return (64,)
    
    def _infer_input_dim(self, block: nn.Module):
        """Infer the input feature dimension for the block."""
        for module in block.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 64  # Default fallback

    # ---------- helpers (all vectorized) ----------

    @staticmethod
    def _flatten_features(t: torch.Tensor) -> torch.Tensor:
        if t.dim() <= 2:
            return t
        return t.flatten(start_dim=2)

    def _scores_to_weights(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (mc_samples, B) scalar scores
        returns: (mc_samples, B) weights, sum over samples == 1 per batch
        """
        logits = s / self.temperature
        w = torch.softmax(logits, dim=0)
        return w

    @staticmethod
    def _weighted_mean(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        while w.dim() < y.dim():
            w = w.unsqueeze(-1)
        return (w * y).sum(dim=0)

    @staticmethod
    def _weighted_var(y: torch.Tensor, w: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        S = y.shape[0]
        if S <= 1:
            return torch.zeros_like(mu)

        while w.dim() < y.dim():
            w = w.unsqueeze(-1)
        diff2 = (y - mu.unsqueeze(0)) ** 2
        num = (w * diff2).sum(dim=0)

        w2sum = (w.squeeze(-1) ** 2).sum(dim=0)
        while w2sum.dim() < mu.dim():
            w2sum = w2sum.unsqueeze(-1)

        denom = (1.0 - w2sum).clamp_min(1e-12)
        return num / denom

    @staticmethod
    def _weighted_quantiles(
        y: torch.Tensor, w: torch.Tensor, qs: torch.Tensor
    ) -> torch.Tensor:
        """
        y: (S,B,...)  predictions
        w: (S,B)      nonnegative, sum over S = 1 per B
        qs: (Q,)      quantiles in [0,1]
        returns: (Q, B, ...) weighted quantiles
        """
        S, B = y.shape[:2]
        feat_shape = y.shape[2:]

        y_flat = y.reshape(S, B, -1)
        y_perm = y_flat.permute(1, 2, 0)
        y_sorted, sort_idx = torch.sort(y_perm, dim=-1)

        w_rep = w.unsqueeze(-1).expand(S, B, y_flat.shape[-1])
        w_rep = w_rep.permute(1, 2, 0)
        w_sorted = torch.gather(w_rep, dim=-1, index=sort_idx)

        cumsum_w = torch.cumsum(w_sorted, dim=-1)
        total_w = cumsum_w[..., -1:].clamp_min(1e-12)
        cdf = cumsum_w / total_w

        Q = qs.numel()
        qs_exp = qs.view(1, 1, Q).to(cdf.device, cdf.dtype)
        cdf_exp = cdf.unsqueeze(-1).expand(-1, -1, -1, Q)
        mask = (cdf_exp >= qs_exp)
        
        mask_int = mask.to(torch.int32)
        first_idx = torch.argmax(mask_int, dim=2)
        prev_idx = (first_idx - 1).clamp_min(0)

        b_idx = torch.arange(B, device=y.device).view(B, 1, 1).expand(B, y_flat.shape[-1], Q)
        k_idx = torch.arange(y_flat.shape[-1], device=y.device).view(1, -1, 1).expand(B, y_flat.shape[-1], Q)

        cdf_prev = cdf[b_idx, k_idx, prev_idx]
        cdf_next = cdf[b_idx, k_idx, first_idx].clamp_min(1e-12)

        y_prev = y_sorted[b_idx, k_idx, prev_idx]
        y_next = y_sorted[b_idx, k_idx, first_idx]

        t = ((qs_exp.expand_as(cdf_prev) - cdf_prev) / (cdf_next - cdf_prev).clamp_min(1e-12)).clamp(0, 1)
        q_vals = y_prev + t * (y_next - y_prev)

        q_vals = q_vals.permute(2, 0, 1)
        q_vals = q_vals.reshape(Q, B, *feat_shape)
        return q_vals

    def forward(
        self,
        x: torch.Tensor,
        return_stats: Tuple[str, ...] = ("mean", "std", "confidence"),
        quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
        confidence_method: Literal["inverse_cv", "inverse_std", "agreement", "sharpness"] = "inverse_cv",
    ) -> Dict[str, torch.Tensor]:

        B = x.shape[0]

        # (1) replicate input across samples
        x_rep = x.repeat(self.mc_samples, *([1] * (x.dim() - 1)))
        # (2) apply dropout (unscaled multiplicative masking)
        mask = torch.bernoulli(torch.full_like(x_rep, 1 - self.dropout_p))
        x_drop = x_rep * mask

        # (3) main block forward
        y_merged = self.block(x_drop)
        out_shape = y_merged.shape[1:]
        y = y_merged.view(self.mc_samples, B, *out_shape)  # (S,B,...)

        # (4) compute weights based on mode
        if self.reweight:
            if self.attention_mode == 'auxiliary':
                # Use auxiliary network for scoring
                if self.attend_mode == 'inputs':
                    # Score on dropped inputs
                    attend_data = x_drop
                else:  # 'outputs'
                    # Score on model outputs
                    attend_data = y_merged
                
                aux_out = self.aux_net(attend_data)
                aux_out = aux_out.view(self.mc_samples, B, -1)
                
                if self.score_reduce == 'mean':
                    s = aux_out.mean(dim=-1)
                elif self.score_reduce == 'sum':
                    s = aux_out.sum(dim=-1)
                elif self.score_reduce == 'l2':
                    s = torch.linalg.vector_norm(aux_out, ord=2, dim=-1)
                elif self.score_reduce == 'logsumexp':
                    s = torch.logsumexp(aux_out, dim=-1)
                else:
                    raise ValueError(f"Unknown score_reduce: {self.score_reduce}")

                w = self._scores_to_weights(s)
                
            elif self.attention_mode == 'attention':
                # attention over MC samples
                if self.attend_mode == 'inputs':
                    # Attend on dropped inputs
                    x_drop_reshaped = x_drop.view(self.mc_samples, B, -1)
                    w = self.cross_attn(x_drop_reshaped)  # (S, B)
                else:  # 'outputs'
                    # Attend on model outputs
                    w = self.cross_attn(y)  # (S, B)
                
        else:
            # uniform weights for plain MC
            w = torch.full((self.mc_samples, B), 1.0 / self.mc_samples, 
                          dtype=y.dtype, device=y.device)

        # (5) compute requested statistics (weighted or unweighted)
        results: Dict[str, torch.Tensor] = {}

        if "samples" in return_stats:
            results["samples"] = y

        # mean / median
        if "mean" in return_stats or "cv" in return_stats or "confidence" in return_stats:
            mu = self._weighted_mean(y, w)
            if "mean" in return_stats:
                results["mean"] = mu
        else:
            mu = None

        if "median" in return_stats:
            qs_t = torch.tensor([0.5], device=y.device, dtype=y.dtype)
            med = self._weighted_quantiles(y, w, qs_t)[0]
            results["median"] = med

        # dispersion
        if "std" in return_stats or "variance" in return_stats or "cv" in return_stats or "confidence" in return_stats:
            if mu is None:
                mu = self._weighted_mean(y, w)
            var = self._weighted_var(y, w, mu)
            std = torch.sqrt(var.clamp_min(0))
            if "variance" in return_stats:
                results["variance"] = var
            if "std" in return_stats:
                results["std"] = std
        else:
            std = None

        if "iqr" in return_stats:
            q = torch.tensor([0.25, 0.75], device=y.device, dtype=y.dtype)
            q25, q75 = self._weighted_quantiles(y, w, q)
            results["iqr"] = (q75 - q25)

        if "range" in return_stats:
            results["range"] = y.max(dim=0).values - y.min(dim=0).values

        if "cv" in return_stats:
            eps = 1e-10
            results["cv"] = (std / (torch.abs(mu) + eps)) if std is not None else torch.zeros_like(mu)

        if "quantiles" in return_stats:
            q = torch.tensor(quantiles, device=y.device, dtype=y.dtype)
            results["quantiles"] = self._weighted_quantiles(y, w, q)

        if "confidence" in return_stats:
            eps = 1e-10
            if confidence_method == "inverse_cv":
                cv = std / (torch.abs(mu) + eps) if std is not None else torch.zeros_like(mu)
                results["confidence"] = torch.exp(-cv)
            elif confidence_method == "inverse_std":
                results["confidence"] = torch.exp(-std) if std is not None else torch.ones_like(mu)
            elif confidence_method == "agreement":
                rng = (y.max(dim=0).values - y.min(dim=0).values) + eps
                norm_std = (std / rng) if std is not None else torch.zeros_like(mu)
                results["confidence"] = 1.0 - torch.clamp(norm_std, 0, 1)
            elif confidence_method == "sharpness":
                if mu.dim() > 1:
                    mean_mag = torch.sqrt((mu ** 2).mean(dim=-1, keepdim=True))
                else:
                    mean_mag = torch.abs(mu)
                rel = (std / (mean_mag + eps)) if std is not None else torch.zeros_like(mu)
                results["confidence"] = torch.exp(-rel)
            else:
                raise ValueError(
                    f"Unknown confidence_method: '{confidence_method}'. "
                    f"Must be one of: 'inverse_cv', 'inverse_std', 'agreement', 'sharpness'"
                )
            
        if 'weights' in return_stats:
            results['weights'] = w

        return results

    def get_mean_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_stats=("mean",))["mean"]

    def get_mean_and_confidence(
        self,
        x: torch.Tensor,
        confidence_method: Literal["inverse_cv", "inverse_std", "agreement", "sharpness"] = "inverse_cv",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, return_stats=("mean", "confidence"), confidence_method=confidence_method)
        return out["mean"], out["confidence"]

    def extra_repr(self) -> str:
        base = (f"dropout_p={self.dropout_p}, mc_samples={self.mc_samples}, "
                f"reweight={self.reweight}")
        if self.reweight:
            base += f", attention_mode='{self.attention_mode}', attend_mode='{self.attend_mode}'"
            if self.attention_mode == 'auxiliary':
                base += f", score_reduce='{self.score_reduce}', temperature={self.temperature}"
            elif self.attention_mode == 'attention':
                base += f", num_heads={self.num_heads}, embedding_size={self.embedding_size}"
                base += f", query_mode='{self.query_mode}'"
        return base

class AttentionScorer(nn.Module):
    """
    attention over MC samples implemented with nn.MultiheadAttention.

    Inputs
    ------
    y : Tensor of shape (S, B, D)
        MC predictions: S samples, batch B, feature dim D.

    Behavior
    --------
    - query_mode='mean':
        * Query is a single token per batch: the mean over S samples.
        * Keys/Values are the S samples.
        * The attention distribution is (per batch) over the S samples.
        * We return the (softmax) attention weights with shape (S, B).

    - query_mode='per_sample':
        * Queries are each of the S samples (self-attention over samples).
        * We get attention probs of shape (B, num_heads, S, S).
        * We average over queries (axis=2) AND heads (axis=1) to get one
          distribution over S per batch, then transpose to (S, B).

    Notes
    -----
    - If embedding_size is provided and != feature_dim, we project y via a
      learnable linear layer to 'embedding_size' before feeding MHA.
    - We always aggregate *probabilities* (not logits) to preserve proper
      normalization semantics.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        embedding_size: Optional[int] = None,
        query_mode: Literal["mean", "per_sample"] = "mean",
        layer_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.query_mode = query_mode

        # Choose working dimension for MHA
        self.embed_dim = int(embedding_size) if embedding_size is not None else self.feature_dim

        # Optional input projection to match MHA embed_dim
        if self.embed_dim != self.feature_dim:
            self.input_proj = nn.Linear(self.feature_dim, self.embed_dim)
        else:
            self.input_proj = nn.Identity()

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        # Use batch_first=False so tensors are (seq_len, batch, embed_dim) like classic MHA
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=False)
        self.layer_norm = nn.LayerNorm(self.feature_dim) if layer_norm else nn.Identity()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y : Tensor (S, B, D)

        Returns
        -------
        weights : Tensor (S, B)
            Normalized attention weights over the S samples, per batch element.
        """
        S, B, D = y.shape
        if D != self.feature_dim:
            raise ValueError(f"y has feature dim D={D}, expected {self.feature_dim}")

        # Project to embed_dim if requested
        y = self.layer_norm(y)
        y_proj = self.input_proj(y)  # (S, B, embed_dim)

        if self.query_mode == "mean":
            # Single query token per batch = mean over samples.
            # query: (1, B, E), key/value: (S, B, E)
            q = y_proj.mean(dim=0, keepdim=True)  # (1, B, E)
            k = y_proj
            v = y_proj

            # average_attn_weights=True -> attn_w shape (B, T=1, S) averaged over heads
            _, attn_w = self.mha(q, k, v, need_weights=True, average_attn_weights=True)
            # Convert to (S, B)
            w = attn_w[:, 0, :].transpose(0, 1)  # (S, B)

        elif self.query_mode == "per_sample":
            # Self-attention over the S samples.
            # query/key/value: (S, B, E)
            q = y_proj
            k = y_proj
            v = y_proj

            # We want per-head probs to average them ourselves over queries and heads.
            # average_attn_weights=False -> (B, num_heads, T=S, S)
            _, attn_w = self.mha(q, k, v, need_weights=True, average_attn_weights=False)
            # Average over queries (axis=2) and heads (axis=1) -> (B, S)
            w_bs = attn_w.mean(dim=(1, 2))  # (B, S)
            # Convert to (S, B)
            w = w_bs.transpose(0, 1)  # (S, B)

        else:
            raise ValueError(f"Unknown query_mode: {self.query_mode}. Use 'mean' or 'per_sample'.")

        # 'w' is already softmax-normalized over S (per batch) by MHA semantics.
        return w


# ============================================================================
# Test Suite
# ============================================================================

def test_all_configurations():
    """Comprehensive test of all configuration combinations"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CONFIGURATION TESTS")
    print("=" * 80)
    
    # Simple test block with output dim divisible by common num_heads values
    block = nn.Sequential(
        nn.Linear(12, 24),  # Input 12 (divisible by 2,3,4,6)
        nn.ReLU(),
        nn.Linear(24, 8)    # Output 8 (divisible by 2,4)
    )
    
    test_cases = [
        # (name, config_dict)
        ("No reweighting (plain MC)", {
            "reweight": False
        }),
        ("Auxiliary on outputs", {
            "reweight": True,
            "attention_mode": "auxiliary",
            "attend_mode": "outputs"
        }),
        ("Auxiliary on inputs", {
            "reweight": True,
            "attention_mode": "auxiliary",
            "attend_mode": "inputs"
        }),
        ("Cross-attn (mean query) on outputs", {
            "reweight": True,
            "attention_mode": "attention",
            "attend_mode": "outputs",
            "query_mode": "mean",
            "num_heads": 4
        }),
        ("Cross-attn (mean query) on inputs", {
            "reweight": True,
            "attention_mode": "attention",
            "attend_mode": "inputs",
            "query_mode": "mean",
            "num_heads": 4
        }),
        ("Cross-attn (per_sample query) on outputs", {
            "reweight": True,
            "attention_mode": "attention",
            "attend_mode": "outputs",
            "query_mode": "per_sample",
            "num_heads": 4
        }),
        ("Cross-attn (per_sample query) on inputs", {
            "reweight": True,
            "attention_mode": "attention",
            "attend_mode": "inputs",
            "query_mode": "per_sample",
            "num_heads": 4
        }),
    ]
    
    x = torch.randn(4, 12)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Testing {len(test_cases)} configurations...\n")
    
    results_summary = []
    
    for test_name, config in test_cases:
        print(f"Testing: {test_name}")
        print("-" * 80)
        
        # Create fresh block copy
        block_copy = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        block_copy.load_state_dict(block.state_dict())
        
        try:
            # Create layer with config
            mc_layer = MonteCarloLayer(
                block=block_copy,
                dropout_p=0.3,
                mc_samples=32,
                **config
            )
            
            # Forward pass
            results = mc_layer(
                x,
                return_stats=("mean", "std", "confidence", "weights"),
                confidence_method="inverse_cv"
            )
            
            # Verify outputs
            assert "mean" in results, "Missing 'mean' in results"
            assert "std" in results, "Missing 'std' in results"
            assert "confidence" in results, "Missing 'confidence' in results"
            assert "weights" in results, "Missing 'weights' in results"
            
            # Check shapes
            assert results["mean"].shape == (4, 8), f"Wrong mean shape: {results['mean'].shape}"
            assert results["std"].shape == (4, 8), f"Wrong std shape: {results['std'].shape}"
            assert results["confidence"].shape == (4, 8), f"Wrong confidence shape: {results['confidence'].shape}"
            assert results["weights"].shape == (32, 4), f"Wrong weights shape: {results['weights'].shape}"
            
            # Check weight properties
            weights = results["weights"]
            weight_sums = weights.sum(dim=0)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
                f"Weights don't sum to 1: {weight_sums}"
            
            # Summary stats
            mean_val = results["mean"][0].mean().item()
            std_val = results["std"][0].mean().item()
            conf_val = results["confidence"][0].mean().item()
            weight_entropy = -(weights[:, 0] * torch.log(weights[:, 0] + 1e-10)).sum().item()
            
            print(f"✓ Test passed")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std:  {std_val:.4f}")
            print(f"  Conf: {conf_val:.4f}")
            print(f"  Weight entropy: {weight_entropy:.4f}")
            print(f"  Representation: {mc_layer.extra_repr()}")
            
            results_summary.append({
                "name": test_name,
                "status": "PASS",
                "mean": mean_val,
                "std": std_val,
                "conf": conf_val,
                "entropy": weight_entropy
            })
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results_summary.append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })
        
        print()
    
    # Print summary table
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\n{'Configuration':<45} {'Status':<8} {'Mean':<10} {'Std':<10} {'Entropy':<10}")
    print("-" * 90)
    
    for result in results_summary:
        if result["status"] == "PASS":
            print(f"{result['name']:<45} {result['status']:<8} {result['mean']:>9.4f} {result['std']:>9.4f} {result['entropy']:>9.4f}")
        else:
            print(f"{result['name']:<45} {result['status']:<8} ERROR: {result['error']}")
    
    # Overall result
    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    total = len(results_summary)
    print("\n" + "=" * 80)
    print(f"Overall: {passed}/{total} tests passed")
    print("=" * 80)
    
    return all(r["status"] == "PASS" for r in results_summary)


def test_attend_mode_differences():
    """Test that attend_mode='inputs' vs 'outputs' produces different results"""
    print("\n" + "=" * 80)
    print("TEST: attend_mode DIFFERENCES")
    print("=" * 80)
    
    block = nn.Sequential(
        nn.Linear(12, 24),
        nn.ReLU(),
        nn.Linear(24, 8)
    )
    
    x = torch.randn(4, 12)
    
    for attention_mode in ['auxiliary', 'attention']:
        print(f"\nTesting {attention_mode} mode:")
        print("-" * 80)
        
        # Create two layers with same initialization but different attend_mode
        block_inputs = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        block_inputs.load_state_dict(block.state_dict())
        
        block_outputs = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        block_outputs.load_state_dict(block.state_dict())
        
        config = {
            "dropout_p": 0.3,
            "mc_samples": 32,
            "reweight": True,
            "attention_mode": attention_mode,
        }
        
        if attention_mode == 'attention':
            config["num_heads"] = 4
            config["query_mode"] = "mean"
        
        # Layer attending on inputs
        layer_inputs = MonteCarloLayer(
            block=block_inputs,
            attend_mode="inputs",
            **config
        )
        
        # Layer attending on outputs
        layer_outputs = MonteCarloLayer(
            block=block_outputs,
            attend_mode="outputs",
            **config
        )
        
        # Get results
        with torch.no_grad():
            results_inputs = layer_inputs(x, return_stats=("mean", "weights"))
            results_outputs = layer_outputs(x, return_stats=("mean", "weights"))
        
        # Compare weights
        weights_inputs = results_inputs["weights"]
        weights_outputs = results_outputs["weights"]
        
        # Check that weights are different
        weights_diff = (weights_inputs - weights_outputs).abs().mean().item()
        
        print(f"  Average weight difference: {weights_diff:.6f}")
        
        if weights_diff > 1e-4:
            print(f"  ✓ Weights are different (as expected)")
        else:
            print(f"  ⚠ Weights are very similar (unexpected)")
        
        # Show weight statistics
        print(f"\n  Weights when attending on inputs:")
        print(f"    Entropy: {-(weights_inputs[:, 0] * torch.log(weights_inputs[:, 0] + 1e-10)).sum().item():.4f}")
        print(f"    Max weight: {weights_inputs[:, 0].max().item():.6f}")
        print(f"    Min weight: {weights_inputs[:, 0].min().item():.6f}")
        
        print(f"\n  Weights when attending on outputs:")
        print(f"    Entropy: {-(weights_outputs[:, 0] * torch.log(weights_outputs[:, 0] + 1e-10)).sum().item():.4f}")
        print(f"    Max weight: {weights_outputs[:, 0].max().item():.6f}")
        print(f"    Min weight: {weights_outputs[:, 0].min().item():.6f}")


def test_backward_pass():
    """Test that gradients flow correctly through all modes"""
    print("\n" + "=" * 80)
    print("TEST: GRADIENT FLOW")
    print("=" * 80)
    
    configs = [
        ("No reweighting", {"reweight": False}),
        ("Auxiliary on inputs", {"reweight": True, "attention_mode": "auxiliary", "attend_mode": "inputs"}),
        ("Auxiliary on outputs", {"reweight": True, "attention_mode": "auxiliary", "attend_mode": "outputs"}),
        ("Cross-attn on inputs", {"reweight": True, "attention_mode": "attention", 
                                  "attend_mode": "inputs", "num_heads": 4}),
        ("Cross-attn on outputs", {"reweight": True, "attention_mode": "attention", 
                                   "attend_mode": "outputs", "num_heads": 4}),
    ]
    
    x = torch.randn(4, 12, requires_grad=True)
    target = torch.randn(4, 8)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Target shape: {target.shape}\n")
    
    for name, config in configs:
        print(f"Testing: {name}")
        print("-" * 80)
        
        block = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        
        mc_layer = MonteCarloLayer(
            block=block,
            dropout_p=0.3,
            mc_samples=16,
            **config
        )
        
        # Forward pass
        results = mc_layer(x, return_stats=("mean",))
        pred = results["mean"]
        
        # Compute loss
        loss = F.mse_loss(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        has_input_grad = x.grad is not None
        has_block_grad = any(p.grad is not None for p in block.parameters())
        
        if config.get("reweight"):
            if config["attention_mode"] == "auxiliary":
                has_aux_grad = any(p.grad is not None for p in mc_layer.aux_net.parameters())
            else:  # attention
                has_aux_grad = any(p.grad is not None for p in mc_layer.cross_attn.parameters())
        else:
            has_aux_grad = None
        
        print(f"  Input gradient: {'✓' if has_input_grad else '✗'}")
        print(f"  Block gradient: {'✓' if has_block_grad else '✗'}")
        if has_aux_grad is not None:
            print(f"  Reweight module gradient: {'✓' if has_aux_grad else '✗'}")
        
        # Compute gradient norm
        if has_input_grad:
            input_grad_norm = x.grad.norm().item()
            print(f"  Input gradient norm: {input_grad_norm:.6f}")
        
        if has_block_grad:
            block_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in block.parameters() if p.grad is not None)).item()
            print(f"  Block gradient norm: {block_grad_norm:.6f}")
        
        # Clear gradients for next test
        x.grad = None
        for p in mc_layer.parameters():
            if p.grad is not None:
                p.grad = None
        
        print()
    
    print("=" * 80)
    print("Gradient flow test complete")
    print("=" * 80)


def demo_basic_usage():
    """Demo 1: Basic MC Dropout with uniform weighting"""
    print("=" * 80)
    print("DEMO 1: Basic MC Dropout (Uniform Weighting)")
    print("=" * 80)
    
    # Simple regression model
    block = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Wrap with MC Dropout (no reweighting)
    mc_layer = MonteCarloLayer(
        block=block,
        dropout_p=0.3,
        mc_samples=32,
        reweight=False  # Uniform weights
    )
    
    # Create sample input
    x = torch.randn(4, 10)  # Batch of 4
    
    print(f"\nInput shape: {x.shape}")
    print(f"MC samples: {mc_layer.mc_samples}")
    print(f"Dropout prob: {mc_layer.dropout_p}")
    
    # Forward pass with various statistics
    results = mc_layer(
        x,
        return_stats=("mean", "std", "confidence", "samples"),
        confidence_method="inverse_cv"
    )
    
    print(f"\nOutput shapes:")
    print(f"  Mean:       {results['mean'].shape}")
    print(f"  Std:        {results['std'].shape}")
    print(f"  Confidence: {results['confidence'].shape}")
    print(f"  Samples:    {results['samples'].shape}")
    
    print(f"\nSample statistics (first batch element):")
    print(f"  Mean:       {results['mean'][0].item():.4f}")
    print(f"  Std:        {results['std'][0].item():.4f}")
    print(f"  Confidence: {results['confidence'][0].item():.4f}")
    print(f"  Sample range: [{results['samples'][:, 0].min().item():.4f}, "
          f"{results['samples'][:, 0].max().item():.4f}]")


def demo_auxiliary_attention():
    """Demo 2: MC Dropout with auxiliary network scoring"""
    print("\n" + "=" * 80)
    print("DEMO 2: Auxiliary Network Importance Sampling")
    print("=" * 80)
    
    # Main prediction block
    block = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Smaller auxiliary network for scoring
    aux_net = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # MC Dropout with auxiliary scoring on inputs
    mc_layer = MonteCarloLayer(
        block=block,
        dropout_p=0.4,
        mc_samples=64,
        reweight=True,
        attention_mode='auxiliary',
        attend_mode='inputs',
        aux_net=aux_net,
        score_reduce='mean',
        temperature=0.5  # Sharper weighting
    )
    
    x = torch.randn(8, 10)
    
    print(f"\nInput shape: {x.shape}")
    print(f"MC samples: {mc_layer.mc_samples}")
    print(f"Attention mode: {mc_layer.attention_mode}")
    print(f"Attend mode: {mc_layer.attend_mode}")
    print(f"Temperature: {mc_layer.temperature}")
    
    # Get results with weights
    results = mc_layer(
        x,
        return_stats=("mean", "std", "confidence", "weights"),
        confidence_method="sharpness"
    )
    
    # Analyze weight distribution
    weights = results['weights']  # (mc_samples, batch)
    print(f"\nWeight statistics (first batch element):")
    print(f"  Shape: {weights.shape}")
    print(f"  Min weight:    {weights[:, 0].min().item():.6f}")
    print(f"  Max weight:    {weights[:, 0].max().item():.6f}")
    print(f"  Mean weight:   {weights[:, 0].mean().item():.6f}")
    print(f"  Weight entropy: {-(weights[:, 0] * torch.log(weights[:, 0] + 1e-10)).sum().item():.4f}")
    print(f"  Effective samples: {(1 / (weights[:, 0] ** 2).sum()).item():.2f} / {mc_layer.mc_samples}")
    
    print(f"\nPrediction statistics:")
    print(f"  Mean ± Std: {results['mean'][0].item():.4f} ± {results['std'][0].item():.4f}")
    print(f"  Confidence: {results['confidence'][0].item():.4f}")


def demo_attention():
    """Demo 3: MC Dropout with attention scoring"""
    print("\n" + "=" * 80)
    print("DEMO 3: attention over MC Samples")
    print("=" * 80)
    
    # Block with multi-dimensional output
    block = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32)  # 32-dim output
    )
    
    x = torch.randn(5, 10)
    
    attend_modes = ['inputs', 'outputs']
    
    for attend_mode in attend_modes:
        print(f"\n{'─' * 80}")
        print(f"Attend Mode: '{attend_mode}'")
        print('─' * 80)
        
        # Create fresh block copy
        block_copy = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        block_copy.load_state_dict(block.state_dict())
        
        # MC Dropout with attention
        mc_layer = MonteCarloLayer(
            block=block_copy,
            dropout_p=0.3,
            mc_samples=48,
            reweight=True,
            attention_mode='attention',
            attend_mode=attend_mode,
            num_heads=4,
            embedding_size=32,
            query_mode='mean'
        )
        
        print(f"\nInput shape: {x.shape}")
        print(f"MC samples: {mc_layer.mc_samples}")
        print(f"Num heads: {mc_layer.num_heads}")
        print(f"Attend mode: {mc_layer.attend_mode}")
        
        results = mc_layer(
            x,
            return_stats=("mean", "std", "variance", "confidence", "weights"),
            confidence_method="agreement"
        )
        
        print(f"\nOutput shapes:")
        print(f"  Mean:     {results['mean'].shape}")
        print(f"  Std:      {results['std'].shape}")
        print(f"  Variance: {results['variance'].shape}")
        print(f"  Weights:  {results['weights'].shape}")
        
        # Show attention weight distribution
        weights = results['weights'][:, 0]  # First batch element
        top_5_idx = torch.topk(weights, k=5).indices
        print(f"\nTop 5 attended samples (batch element 0):")
        for i, idx in enumerate(top_5_idx):
            print(f"  Sample {idx.item():2d}: weight = {weights[idx].item():.6f}")
        
        # Weight statistics
        print(f"\nWeight statistics:")
        print(f"  Min:     {weights.min().item():.6f}")
        print(f"  Max:     {weights.max().item():.6f}")
        print(f"  Entropy: {-(weights * torch.log(weights + 1e-10)).sum().item():.4f}")


def main():
    """Run all demos and tests"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "MonteCarloLayer Demo & Test Suite" + " " * 25 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    torch.manual_seed(42)  # For reproducibility
    
    try:
        # Run comprehensive tests first
        print("\n" + "█" * 80)
        print("█" + " RUNNING TESTS " + "█" * 64)
        print("█" * 80)
        
        all_passed = test_all_configurations()
        test_attend_mode_differences()
        test_backward_pass()
        
        if not all_passed:
            print("\n⚠ Some tests failed! Check output above.")
            return
        
        # Run demos
        print("\n\n" + "█" * 80)
        print("█" + " RUNNING DEMOS " + "█" * 64)
        print("█" * 80)
        
        demo_basic_usage()
        demo_auxiliary_attention()
        demo_attention()
        
        print("\n" + "=" * 80)
        print("All tests and demos completed successfully! ✓")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()