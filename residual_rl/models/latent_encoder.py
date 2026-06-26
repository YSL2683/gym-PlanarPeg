import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class DINOFrontEncoder(nn.Module):
    """
    DINOv2 backbone for front-view image encoding.
    Expected input: (B, 3, H, W) float32 images in range [0, 1].
    """

    def __init__(self, model_name="dinov2_vits14_reg", device="cuda"):
        super().__init__()
        self.device = device
        self.dino = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        
        # DINOv2 uses ImageNet normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Check output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            dummy_output = self.dino(dummy_input)
            self.embed_dim = dummy_output.shape[-1] # Usually 384 for vits14

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float32, normalized [0, 1]
        Returns:
            embeddings: (B, embed_dim)
        """
        images = self.normalize(images)
        return self.dino(images)

class MLPE2C(nn.Module):
    """
    MLP-based Embed to Control model.
    Learns a latent space with locally-linear transition dynamics
    from pre-computed DINO embeddings.
    """

    def __init__(self, obs_dim=384, action_dim=3, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.obs_dim = obs_dim

        # Encoder: DINO embedding -> latent (z_mean, z_logvar)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, z_dim * 2),
        )

        # Decoder: latent -> reconstructed DINO embedding
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

        # Locally-linear transition model: z' = A(z)z + B(z)a + o(z)
        self.trans_mlp = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, z_dim * (2 + action_dim + 1))
        )
        self.action_dim = action_dim

    def enc(self, x):
        """Encode DINO embedding to (z_mean, z_logvar)."""
        h = self.encoder(x)
        z_mean, z_logvar = h.chunk(2, dim=-1)
        z_logvar = torch.clamp(z_logvar, min=-10.0, max=2.0)
        return z_mean, z_logvar

    def dec(self, z):
        """Decode latent to DINO embedding space."""
        return self.decoder(z)

    def transition(self, z, a):
        """Predict next latent state using locally-linear dynamics."""
        B = z.shape[0]
        trans_params = self.trans_mlp(z)
        
        # Extract components
        idx = 0
        u = trans_params[:, idx:idx+self.z_dim]
        idx += self.z_dim
        v = trans_params[:, idx:idx+self.z_dim]
        idx += self.z_dim
        B_mat_flat = trans_params[:, idx:idx+self.z_dim*self.action_dim]
        idx += self.z_dim*self.action_dim
        offset = trans_params[:, idx:idx+self.z_dim]

        # A = I + u * v^T (rank-1 perturbation)
        # A @ z = z + u * (v^T @ z)
        v_dot_z = (v * z).sum(dim=1, keepdim=True)
        A_z = z + u * v_dot_z

        B_mat = B_mat_flat.view(B, self.z_dim, self.action_dim)
        B_a = torch.bmm(B_mat, a.unsqueeze(-1)).squeeze(-1)

        z_next = A_z + B_a + offset
        return z_next

    def _reparameterize(self, mean, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, obs, action, next_obs):
        """
        Compute all loss components.
        Returns: (kl_loss, recon_loss, transition_consistency_loss)
        """
        z_mean, z_logvar = self.enc(obs)
        z = self._reparameterize(z_mean, z_logvar)

        z_next_mean, z_next_logvar = self.enc(next_obs)
        z_next = self._reparameterize(z_next_mean, z_next_logvar)

        # Reconstruction loss
        next_obs_recon = self.dec(z_next_mean) # use mean for dec
        recon_loss = F.mse_loss(next_obs_recon, next_obs)

        # KL divergence: q(z|obs) vs N(0, I)
        kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(dim=-1).mean()

        # Transition consistency loss
        z_pred_next = self.transition(z_mean, action)
        
        # We model the predicted transition as N(z_pred_next, pred_z_cov)
        # Following LaNE, pred_z_cov is computed by propagating z_cov through A.
        # For simplicity in this implementation, and since we just want the KL between
        # N(z_pred_next, ...) and q(z_{t+1}|obs_{t+1}), we use the simplified KL from LaNE:
        trans_loss = 0.5 * (
            (z_next_mean - z_pred_next).pow(2) / z_next_logvar.exp()
            + z_next_logvar - z_logvar # Approximation
        ).sum(dim=-1).mean()

        return kl_loss, recon_loss, trans_loss
