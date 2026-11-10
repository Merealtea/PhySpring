from core_model import MLP, gumbel_softmax
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm, Softplus
import torch
from torch import nn
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.utils import (
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor,
    coalesce,
)
from core_model import EvoMesh, WeightedEdgeConv, Unpool
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
    
class PhysicsAwareAttentionPooling(nn.Module):
    def __init__(self, in_features: int, num_heads: int = 1, k_eigenvectors: int = 8, 
                 lambda_reg: float = 0.1, sigma: float = 1.0):
        """
        in_features: node input feature dimension
        num_heads: number of attention heads
        k_eigenvectors: use the first k nonzero eigenvectors of the Laplacian matrix as positional encoding
        lambda_reg: regularization weight on the number of clusters (fix: newly added initialization)
        sigma: Gaussian kernel bandwidth used to build graph topology from spatial coordinates (fix: newly added initialization)
        """
        super().__init__()
        self.k_eigenvectors = k_eigenvectors
        self.lambda_reg = lambda_reg 
        self.sigma = sigma
        
        # 1. Linear projection layer for positional encoding
        self.pe_proj = nn.Linear(k_eigenvectors, in_features)
        
        # 2. PyTorch native MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=in_features, 
            num_heads=num_heads, 
            batch_first=True
        )

    def compute_laplacian_from_pos(self, node_pos: torch.Tensor, adj_mat: torch.Tensor = None) -> torch.Tensor:
        """
        Dynamically compute the physical Laplacian matrix from node coordinates.
        Supports passing adj_mat for topology-level forced sparsification.
        """
        N = node_pos.shape[-2]
        D_pos = node_pos.shape[-1]
        device, dtype = node_pos.device, node_pos.dtype
        
        # 1. Deterministic perturbation to prevent coincident points from crashing the eigenvalue solve
        idx = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        dim_idx = torch.arange(1, D_pos + 1, device=device, dtype=dtype).unsqueeze(0)
        deterministic_noise = torch.sin(idx * dim_idx) * 1e-5
        stable_pos = node_pos + deterministic_noise
        
        # 2. Compute spatial distances and apply the Gaussian kernel
        dist_matrix = torch.cdist(stable_pos, stable_pos, p=2.0)
        A = torch.exp(-(dist_matrix ** 2) / (2 * self.sigma ** 2))
        A = A * adj_mat.float()
        
        # 3. Zero out the diagonal (remove self-loops)
        A.diagonal(dim1=-2, dim2=-1).zero_()
        
        # 4. Compute the degree matrix and Laplacian matrix
        degrees = torch.sum(A, dim=-1)
        D = torch.diag_embed(degrees)
        L = D - A
        
        return L

    def compute_pe(self, L: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian positional encoding (fully batch-compatible, explicitly extracts the Top-K smallest nonzero eigenvectors)"""
        # 1. Solve for eigenvalues and eigenvectors
        # eigh is in ascending order by default; eigenvalues: (..., N), eigenvectors: (..., N, N)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # 2. Build a penalty matrix: replace invalid eigenvalues (<= eps) with infinity
        # This way, when searching for the "smallest" eigenvalues, they are automatically pushed to the end and never selected
        eps = 1e-5
        safe_eigenvalues = torch.where(
            eigenvalues > eps, 
            eigenvalues, 
            torch.tensor(float('inf'), device=L.device, dtype=eigenvalues.dtype)
        )
        
        # 3. Explicitly get the indices of the first k smallest eigenvalues (amplitude ordering)
        k = self.k_eigenvectors
        N = L.shape[-1]
        actual_k = min(k, N)
        
        # largest=False means taking from smallest to largest; sorted=True guarantees absolutely correct order
        # topk_indices shape: (..., actual_k)
        _, topk_indices = torch.topk(safe_eigenvalues, k=actual_k, dim=-1, largest=False, sorted=True)
        
        # 4. Dimension-adaptive expansion and eigenvector Gather
        if L.dim() == 3:  # Batch mode: L has shape (B, N, N)
            # Expand indices to (B, N, actual_k) to match eigenvectors (B, N, N)
            gather_indices = topk_indices.unsqueeze(1).expand(-1, N, -1)
            pe = torch.gather(eigenvectors, dim=-1, index=gather_indices)
            
            # Cleanup: if a graph has fewer than k valid eigenvalues, force the eigenvectors corresponding to inf to zero
            valid_mask = torch.gather(safe_eigenvalues, dim=-1, index=topk_indices) < float('inf')
            pe = pe * valid_mask.unsqueeze(1).float() # expand valid_mask to (B, 1, actual_k)
            
        else:  # Single-graph mode: L has shape (N, N)
            gather_indices = topk_indices.unsqueeze(0).expand(N, -1)
            pe = torch.gather(eigenvectors, dim=-1, index=gather_indices)
            
            valid_mask = torch.gather(safe_eigenvalues, dim=-1, index=topk_indices) < float('inf')
            pe = pe * valid_mask.unsqueeze(0).float()
            
        # 5. If the number of nodes N < k, pad the last dimension to size k
        if actual_k < k:
            pad_size = k - actual_k
            pe = F.pad(pe, (0, pad_size), mode='constant', value=0.0)
            
        return pe
    
    def forward(self, x: torch.Tensor, node_pos: torch.Tensor, node_mass: torch.Tensor, node_type: torch.Tensor,
                gs: torch.Tensor, tau: float = 1.0, prev_P = None, object_radius: float = None):
        """
        Inputs:
        x: (N, in_features) node input features
        node_pos: (N, D_pos) node spatial coordinates
        node_mass: (N,) or (N, 1) node masses
        node_type: (N,) node types; there are 4 types, the 4th being control points. We do not merge control points for now; only other points are merged
        adj_mask: (N, N) adjacency mask; i and j are neighbors and can be merged only when adj_mask[i,j]=1
        tau: Gumbel-softmax temperature parameter
        prev_P: precomputed projection matrix; used directly if provided, otherwise computed with gumbel softmax
        object_radius: object radius threshold; when the distance between two points exceeds this value, they are not allowed to merge
        """
        N = x.shape[1]
        control_node_type = 3

        object_radius /= 2 # prevent new edges from exceeding object_radius

        # Build the adjacency matrix from edge index gs
        # gs shape: (2, num_edges), where gs[0] is the source node and gs[1] is the target node
        num_nodes = node_pos.shape[-2]
        adj_mask = torch.zeros(num_nodes, num_nodes, device=node_pos.device, dtype=torch.bool)
        if gs.shape[1] > 0:
            adj_mask[gs[0], gs[1]] = True
            # Make the adjacency matrix symmetric (undirected graph)
            adj_mask = adj_mask | adj_mask.T
        
        # Set the diagonal to False (remove self-loops)
        adj_mask.fill_diagonal_(False)
        
        # ==========================================
        # ★ Step 0: distance constraint based on object_radius (added) ★
        # ==========================================
        # If object_radius is provided, compute distances between all point pairs and update adj_mask
        if object_radius is not None and object_radius > 0:
            # Compute the distance matrix (N, N)
            dist_matrix = torch.cdist(node_pos, node_pos, p=2.0)  # shape: (1, N, N) or (N, N)
            
            # Create the distance constraint mask: positions with distance greater than object_radius are True, meaning merging is not allowed
            distance_forbidden_mask = dist_matrix > object_radius
            
            # If dist_matrix has a batch dimension, squeeze it
            if distance_forbidden_mask.dim() == 3:
                distance_forbidden_mask = distance_forbidden_mask[0]  # (N, N)
            
            # Remove overly distant positions from the adjacency mask
            adj_mask = adj_mask & (~distance_forbidden_mask)
            print(f"[Distance Constraint] object_radius={object_radius}, filtered out {(dist_matrix > object_radius).sum().item()} pairs with distance > {object_radius}")


        z = x.squeeze(0)  
        z_norm = F.normalize(z, p=2, dim=-1)

        # --- Step 3: distance matrix computation ---
        P_gramian = z_norm @ z_norm.transpose(-1, -2)
        diag_P = torch.diag(P_gramian).unsqueeze(1)
        D_sq = diag_P + diag_P.transpose(0, 1) - 2 * P_gramian
        D_mat_base = torch.sqrt(torch.clamp(D_sq, min=0.0) + 1e-8)

        # calculate adj matrix based on gs
        penalty_value = D_mat_base.max().detach() * 10.0
        D_mat = D_mat_base.masked_fill(~adj_mask, penalty_value)

        # ==========================================
        # ★ Step 4: Gumbel-Softmax dynamic assignment + straight-through estimator ★
        # ==========================================
        # Regardless of whether prev_P is provided, first compute logits and the soft version for gradient backpropagation
        logits = -D_mat 
        
        # Build forbidden_mask to control the merging rules
        forbidden_mask = torch.zeros_like(logits, dtype=torch.bool)
        
        # Rule 1: control points cannot be assigned to others - commented out, rule 5 already implies this constraint
        is_control = (node_type == control_node_type)[0]  # find all control points, shape (N,)
        forbidden_mask[is_control, :] = True
        
        # Rule 2: others cannot be assigned to control points - commented out, rule 5 already implies this constraint
        forbidden_mask[:, is_control] = True
        
        # Rule 3: find nodes connected to controller nodes; these nodes also cannot merge with other points - commented out, rule 5 already implies this constraint
        controller_neighbor_mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
        if adj_mask is not None and gs.shape[1] > 0:
            control_indices = torch.where(is_control)[0]
            for ctrl_idx in control_indices:
                neighbors_of_controller = adj_mask[ctrl_idx]
                controller_neighbor_mask = controller_neighbor_mask | neighbors_of_controller
            controller_neighbor_mask = controller_neighbor_mask & (~is_control)
        
        # Rule 3a: nodes connected to the controller cannot be assigned to other non-neighbor nodes - commented out, rule 5 already implies this constraint
        if adj_mask is not None:
            for i in range(N):
                if controller_neighbor_mask[i]:
                    non_neighbors_of_i = ~adj_mask[i]
                    forbidden_mask[i, non_neighbors_of_i] = True
        
        # Rule 3b: other nodes also cannot be assigned to nodes connected to the controller - commented out, rule 5 already implies this constraint
        for j in range(N):
            if controller_neighbor_mask[j]:
                non_neighbors_of_j = ~adj_mask[:, j]
                forbidden_mask[non_neighbors_of_j, j] = True
        
        # Rule 4: only points with the same node_type can merge - commented out, rule 5 already implies this constraint
        # Build a matrix where [i,j] is True if node_type[i] != node_type[j]
        node_type_1d = node_type[0] if node_type.dim() > 1 else node_type  # (N,)
        type_diff_mask = node_type_1d.unsqueeze(0) != node_type_1d.unsqueeze(1)  # (N, N)
        forbidden_mask = forbidden_mask | type_diff_mask
        # Note: rule 5 already implies this constraint, since two object points necessarily have the same type (both 0)
        
        # Rule 5: only object points (node_type == 0) can merge; no other points can merge
        # Build a matrix where [i,j] is True if i or j is not an object point
        # node_type_1d = node_type[0] if node_type.dim() > 1 else node_type  # (N,)
        # is_object = (node_type_1d == 0)  # (N,) - True for object points
        # # If i or j is not an object point, merging is not allowed
        # not_both_object = ~(is_object.unsqueeze(0) & is_object.unsqueeze(1))  # (N, N)
        # forbidden_mask = forbidden_mask | not_both_object
        
        # Rule 6: allow all points (including control points) to be assigned to themselves (the diagonal is always allowed)
        diag_indices = torch.arange(N, device=logits.device)
        forbidden_mask[diag_indices, diag_indices] = False 
        
        # Set logits of forbidden assignment paths to a very small value so their probability approaches 0 during Gumbel-softmax sampling
        logits = logits.masked_fill(forbidden_mask, -1e9)
        
        # Compute the soft version for gradient backpropagation
        P_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        
        if prev_P is not None:
            # Use the straight-through estimator: forward uses prev_P, backward passes gradients to logits through P_soft
            # The form P_full = prev_P + (P_soft - P_soft).detach() is incorrect
            # Correct straight-through estimator form: P_full = P_soft + (prev_P - P_soft).detach()
            # Then forward gives P_full = prev_P, and backward passes gradients through P_soft
            P_full = P_soft + (prev_P - P_soft).detach()
        else:
            # When prev_P is not used, use standard hard gumbel softmax
            P_full = (P_soft == P_soft.max(dim=-1, keepdim=True)[0]).float()
        
        # Compute r_soft for the loss (computed using the soft version)
        r_soft = P_soft.max(dim=0)[0].sum()

        # --- Step 5: extract the reduced-order projection matrix P_proj ---
        active_mask = P_full.sum(dim=0) > 0
        P_proj = P_full[:, active_mask] # shape: (N, r)
        dynamic_r = P_proj.shape[1]

        # ==========================================
        # ★ Step 6: downsampling of features and positions ★
        # ==========================================
        pos_hat = node_pos[:, active_mask]  
        cluster_sizes = P_proj.sum(dim=0, keepdim=True) 
        P_proj_normalized = P_proj / cluster_sizes.clamp(min=1e-9)
        x_hat = P_proj_normalized.transpose(-1, -2) @ x 

        A_orig = adj_mask.float()

        A_new = P_proj.transpose(-1, -2) @ A_orig @ P_proj
        A_new.fill_diagonal_(0.0)
        
        # Convert A_new to an upper triangular matrix (keep only the upper triangular part)
        A_new = torch.triu(A_new, diagonal=1)
        
        src, dst = torch.where(A_new > 0)
        gs = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0) 

        # --- Step 8: reduced-order physical system and loss computation ---
        # Directly multiply the mass vector by the transpose of P_proj
        # (..., r, N) @ (..., N, 1) -> (..., r, 1)
        node_mass_hat = node_mass @ P_proj

        node_type_hat = node_type[:, active_mask]

        # (fix) lambda_reg is now accessed via self
        intra_cluster_loss = torch.trace(P_proj.transpose(-1, -2) @ D_mat @ P_proj)
        total_loss = (intra_cluster_loss  + self.lambda_reg * r_soft) / N
        
        ids = torch.where(active_mask)[0].cpu().numpy().tolist()

        return ids, gs, x_hat, pos_hat, node_mass_hat, node_type_hat, total_loss, dynamic_r, P_full
    
class End2End_amp_base(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim=8, mass_embedding_dim=8):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        # edge_info_in_len: 2 * latent_dim for node i and j
        # + 2 * pos_dim + 2 for lagrangian (dir and norm_w and norm_m)
        # + 20 for spring_encoded
        # + 2 * node_type_embedding_dim for node type embedding (i and j)
        # + 2 * mass_embedding_dim for node mass embedding (i and j)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 + 20 + 2 * node_type_embedding_dim + 2 * mass_embedding_dim if lagrangian else 2 * latent_dim + pos_dim + 1 + 20 + 2 * node_type_embedding_dim + 2 * mass_embedding_dim
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.mlp_edge_weight = Seq(*[MLP(latent_dim, latent_dim, 1, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim
        
        # Node type embedding: 4 types (0: object, 1: surface, 2: interior, 3: controller)
        self.node_type_embedding = torch.nn.Embedding(4, node_type_embedding_dim)
        
        # Mass embedding: use MLP to embed scalar mass value
        self.mass_embedding = MLP(1, mass_embedding_dim, mass_embedding_dim, 2, True)

    def positional_encoding(self, positions, num_freq_bands=10):
        """
        NeRF-style positional encoding
        Args:
            positions: [N, 3] normalized positions in [-1, 1]
            num_freq_bands: number of frequency bands (L)
        Returns:
            encoded_positions: [N, 3 * 2 * L] encoded positions
        """
        # positions shape: [N, 3]
        freq_bands = 2.0 ** torch.arange(num_freq_bands, dtype=positions.dtype, device=positions.device)  # [L]
        # freq_bands shape: [L]

        # Expand dimensions for broadcasting: positions [N, 3, 1], freq_bands [1, 1, L]
        pos_expanded = positions.unsqueeze(-1)  # [N, 3, 1]
        freq_expanded = freq_bands.unsqueeze(0).unsqueeze(0)  # [1, 1, L]

        # Compute scaled positions: [N, 3, L]
        scaled_pos = math.pi * pos_expanded * freq_expanded

        # Apply sin and cos
        sin_encoded = torch.sin(scaled_pos)  # [N, 3, L]
        cos_encoded = torch.cos(scaled_pos)  # [N, 3, L]

        # Interleave sin and cos: [N, 3, 2*L]
        encoded = torch.stack([sin_encoded, cos_encoded], dim=-1)  # [N, 3, L, 2]
        encoded = encoded.reshape(positions.shape[0], positions.shape[1], -1)  # [N, 3, 2*L]

        # Flatten to [N, 3 * 2 * L]
        encoded = encoded.reshape(positions.shape[0], -1)

        return encoded

    def forward(self, x, g, pos, node_type=None, node_mass=None):
        """
        Args:
            x: node features, shape (T, N, F) or (N, F)
            g: edge index, tuple of (src, dst) tensors
            pos: node positions, shape (T, N, 3) or (N, 3)
            node_type: node types, shape (N,) or (T, N), values in {0, 1, 2, 3}
            node_mass: node masses, shape (N,) or (N, 1) or (T, N, 1)
        """
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim

        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm], dim=-1)

        # normalize spring_rest_length into [-1,1]
        spring_rest_length = fiber[..., 3]
        spring_min = spring_rest_length.min(dim=1, keepdim=True)[0]
        spring_max = spring_rest_length.max(dim=1, keepdim=True)[0]
        normalized_spring_rest_length = (2 * (spring_rest_length - spring_min) / (spring_max - spring_min) - 1)[0]

        # apply NeRF-style positional encoding to spring lengths
        spring_encoded = self.positional_encoding(normalized_spring_rest_length[:, None], num_freq_bands=10)[None]

        # Compute node type and mass embeddings and concatenate to fiber
        if node_type is not None and node_mass is not None:
            # Ensure node_type is 1D
            if node_type.dim() == 2:
                node_type = node_type.squeeze(0)
            # Ensure node_mass is 2D (N, 1)
            if node_mass.dim() == 1:
                node_mass = node_mass.unsqueeze(-1)
            elif node_mass.dim() == 3:
                node_mass = node_mass.squeeze(1)
            
            # Get embeddings for nodes i and j
            node_type_i = self.node_type_embedding(node_type[i])[None]  # (N, type_emb_dim)
            node_type_j = self.node_type_embedding(node_type[j])[None]  # (N, type_emb_dim)
            node_mass_i = self.mass_embedding(node_mass[:, i, None]) # (N, mass_emb_dim)
            node_mass_j = self.mass_embedding(node_mass[:, j, None])  # (N, mass_emb_dim)
            
            # Concatenate embeddings to fiber
            fiber = torch.cat([fiber, node_type_i, node_type_j, node_mass_i, node_mass_j], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j, spring_encoded], dim=-1)
        
        edge_embedding = self.mlp_edge_info(tmp)
        edge_weight = self.mlp_edge_weight(edge_embedding)
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)
        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")
        
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x, edge_weight, edge_embedding

class End2EndReduction(EvoMesh):
    def __init__(self, l_n, pre_l_n, bottom_ln, ld, hidden_layer, pos_dim, lagrangian, enhance=True, agg_conv_pos=False, edge_set_num=1,
                 transformer_hidden_dim=128, transformer_num_layers=2, pos_encoding_dim=30,
                 node_type_embedding_dim=8, mass_embedding_dim=8,
                 pooling_num_heads=1, k_eigenvectors=8, pooling_lambda_reg=0.1, pooling_sigma=1.0):
        super(EvoMesh, self).__init__()
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.downpools = nn.ModuleList()  # use PhysicsAwareAttentionPooling for downsampling
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        self.pre_l_n = pre_l_n
        self.enhance = enhance
        self.agg_conv_pos = agg_conv_pos
        self.bottom_ln = bottom_ln
        self.node_type_embedding_dim = node_type_embedding_dim
        self.mass_embedding_dim = mass_embedding_dim
        self.bottom_gmp = nn.ModuleList(
            End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim, mass_embedding_dim) 
            for _ in range(self.bottom_ln)
        )

        for _ in range(self.l_n):
            self.down_gmps.append(
                End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim, mass_embedding_dim)
            )
            self.downpools.append(
                PhysicsAwareAttentionPooling(ld, pooling_num_heads, k_eigenvectors, pooling_lambda_reg, pooling_sigma)
            )
            self.up_gmps.append(
                End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim, mass_embedding_dim)
            )
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian

        # Transformer components for encoding reduced node features
        self.ld = ld
        self.pos_dim = pos_dim
        self.pos_encoding_dim = pos_encoding_dim
        
        # NeRF-style positional encoding for 3D positions
        # Output dim: 3 * 2 * num_freq_bands = pos_encoding_dim
        self.num_freq_bands = pos_encoding_dim // 6
        
        # Position encoding to transformer hidden dim projection
        self.pos_to_transformer = nn.Linear(self.pos_encoding_dim, transformer_hidden_dim)
        
        # Node feature to transformer hidden dim projection
        self.node_to_transformer = nn.Linear(ld, transformer_hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=max(1, transformer_hidden_dim // 32),
            dim_feedforward=transformer_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        # Transformer output projection back to original feature dim
        self.transformer_output_proj = nn.Linear(transformer_hidden_dim, ld)

        self.edge_weight = MLP(2*ld, ld, ld, hidden_layer, True)
        # Node type embedding: 4 types (0: object, 1: surface, 2: interior, 3: controller)
        self.node_type_embedding = torch.nn.Embedding(4, ld)

    def positional_encoding_3d(self, positions: torch.Tensor) -> torch.Tensor:
        """
        NeRF-style positional encoding for 3D positions
        Args:
            positions: (B, N, 3) or (N, 3) 3D coordinates (normalized to [-1, 1])
        Returns:
            encoded: (B, N, pos_encoding_dim) or (N, pos_encoding_dim)
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, N, D = positions.shape
        device, dtype = positions.device, positions.dtype
        
        # Frequency bands: 2^0, 2^1, ..., 2^(num_freq_bands-1)
        freq_bands = 2.0 ** torch.arange(self.num_freq_bands, dtype=dtype, device=device)  # [L]
        
        # Expand for broadcasting: positions [B, N, 3, 1], freq_bands [1, 1, 1, L]
        pos_expanded = positions.unsqueeze(-1)  # [B, N, 3, 1]
        freq_expanded = freq_bands.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, L]
        
        # Compute scaled positions: [B, N, 3, L]
        scaled_pos = math.pi * pos_expanded * freq_expanded
        
        # Apply sin and cos: [B, N, 3, L]
        sin_encoded = torch.sin(scaled_pos)
        cos_encoded = torch.cos(scaled_pos)
        
        # Stack and reshape: [B, N, 3, L, 2] -> [B, N, 3*L*2]
        encoded = torch.stack([sin_encoded, cos_encoded], dim=-1)  # [B, N, 3, L, 2]
        encoded = encoded.reshape(B, N, -1)  # [B, N, 3*L*2]
        
        if squeeze_output:
            encoded = encoded.squeeze(0)
        
        return encoded

    def _pool_tensor(self, tensor, node_mask):
        tensor = node_mask[None, :, None].float() * tensor
        return tensor[:, node_mask]

    def forward(self, node_in, m_ids, m_gs, m_pos, m_mass, m_type, temp=0.1, m_proj=None, object_radius=None):
        """
        Forward pass that performs internal downsampling with PhysicsAwareAttentionPooling.
        Flow: pool -> down_gmps -> edge conv -> PhysicsAwareAttentionPooling to obtain downsampled gs, proj, etc.
        
        Args:
            node_in: node input features, shape (N, F) or (T, N, F)
            m_ids: list of node indices per layer
            m_gs: list of edge indices per layer
            m_proj: list of projection matrices per layer
            m_node_pos: list of node positions per layer
            m_node_mass: list of node masses per layer
            m_node_type: list of node types per layer
            temp: Gumbel-softmax temperature parameter
            m_proj: list of projection matrices, optional
            object_radius: object radius threshold; when the distance between two points exceeds this value, they are not allowed to merge
        """
        down_ids = []
        down_gs = []
        down_outs = []
        down_ps = []
        down_mass = []
        down_type = []
        cts = []
        pooling_losses = []
        down_proj_list = []  # store the projection matrices generated by PhysicsAwareAttentionPooling
        
        # Ensure the input has a batch dimension
        if node_in.dim() == 2:
            node_in = node_in[None, ...]  # (1, N, F)
        
        # down pass
        ds_edge_embedding = []
        
        gs = m_gs[0]
        pos = m_pos[0][None]
        node_mass = m_mass[0][None]
        node_type = m_type[0][None]

        down_gs = [gs]
        down_ps = [pos]
        down_mass = [node_mass]
        down_type = [node_type]
        
        for i in range(self.l_n):
            # Step 2: down_gmps - process edge information with AMP
            node_in, ew, edge_embedding =\
                self.down_gmps[i](node_in, gs, pos, 
                                    node_type=node_type, 
                                    node_mass=node_mass)

            # Step 3: edge conv - update node features with edge convolution
            node_in = self.edge_conv(node_in, gs, ew)

            down_outs.append(node_in)
            ds_edge_embedding.append(edge_embedding)
            cts.append(ew)
            
            # Step 4: PhysicsAwareAttentionPooling - obtain downsampled gs, proj, etc.
            if ((i+1) > len(m_proj)):
                idx, gs, node_in, pos, node_mass, node_type, pool_loss, dynamic_r, P_proj =\
                    self.downpools[i](
                        node_in, pos, node_mass, node_type, gs, 
                        tau=temp, object_radius=object_radius)
            else:
                idx, gs, node_in, pos, node_mass, node_type, pool_loss, dynamic_r, P_proj =\
                    self.downpools[i](
                        node_in, pos, node_mass, node_type, gs, 
                        temp, m_proj[i], object_radius=object_radius)
                
            down_ids.append(idx)
            down_gs.append(gs)
                
            pooling_losses.append(pool_loss)
            down_ps.append(pos)
            down_mass.append(node_mass)
            down_type.append(node_type)
            down_proj_list.append(P_proj)
            
        # Prepare node_type and node_mass for bottom layer
        for l in range(self.bottom_ln):
            node_in, ew, _ = self.down_gmps[i](node_in, 
                                               down_gs[self.l_n], 
                                               down_ps[self.l_n], 
                                               node_type=node_type, 
                                               node_mass=node_mass)

        # up pass
        mlvl_edge_embedding = []

        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            # Use the gs generated by downpools for the up pass
            gs = down_gs[up_idx] 
            num_edge = gs.shape[1]

            node_in = self.unpools[i](node_in, 
                                      down_outs[up_idx].shape[-2], 
                                      down_ids[up_idx])
            
            node_in= self.edge_conv(node_in, gs, cts[up_idx], aggragating=False)
            # Prepare node_type and node_mass for up pass
            node_type_up = down_type[up_idx]
            node_mass_up = down_mass[up_idx]

            node_in, _, edge_embedding = self.up_gmps[i](
                node_in, gs, down_ps[up_idx], 
                node_type=node_type_up, node_mass=node_mass_up
            )
                        
            node_in = node_in + down_outs[up_idx]

            edge_embedding = edge_embedding + ds_edge_embedding[up_idx]            

            # Average symmetric edges
            edge_feature = edge_embedding[:, :num_edge//2] + edge_embedding[:, num_edge//2:]
            mlvl_edge_embedding.insert(0, edge_feature)
        
        down_ids.insert(0, m_ids[0])
        # Return downsample results for warp construction
        downsample_results = {
            'down_ids': down_ids,
            'down_gs': down_gs,
            'down_proj': down_proj_list,
            'down_ps': down_ps,
            'down_mass': down_mass,
            'down_type': down_type,
        }

        return mlvl_edge_embedding, pooling_losses, downsample_results
