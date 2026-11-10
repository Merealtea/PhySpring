import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import solve_continuous_lyapunov, cholesky
from scipy.cluster.hierarchy import linkage, fcluster
import logging
from time import time
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

def build_physical_matrices_sparse(m_vector, m_gs, Y_vector, d_dashpot, d_drag):
    """
    Build the M, D, L matrices in sparse format, with near-zero memory usage.
    """
    n = len(m_vector)
    
    # 1. Sparse mass matrix M
    M_sparse = sp.diags(m_vector, format='csr')
    
    u = m_gs[0]
    v = m_gs[1]
    valid_mask = (u != v)
    try:
        u_v, v_v, Y_v = u[valid_mask], v[valid_mask], Y_vector[valid_mask]
    except:
        import pdb; pdb.set_trace()
    # 2. Sparse Laplacian matrix L
    rows = np.concatenate([u_v, v_v])
    cols = np.concatenate([v_v, u_v])
    vals = np.concatenate([-Y_v, -Y_v])
    
    L_sparse = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Compute the diagonal: negative of the sum of off-diagonal elements in each row
    diag_L = -np.asarray(L_sparse.sum(axis=1)).flatten()
    L_sparse.setdiag(diag_L + L_sparse.diagonal())
    
    # 3. Sparse damping matrix D
    if d_dashpot > 0 and len(u_v) > 0:
        dash_vals = np.full(len(u_v) * 2, -d_dashpot, dtype=np.float64)
        D_sparse = sp.coo_matrix((dash_vals, (rows, cols)), shape=(n, n)).tocsr()
        diag_D = -np.asarray(D_sparse.sum(axis=1)).flatten()
        D_sparse.setdiag(diag_D + d_drag)
    else:
        D_sparse = sp.diags(np.full(n, d_drag), format='csr')

    return M_sparse, D_sparse, L_sparse


class FastAdaptiveNetworkReducer:
    def __init__(self, distance_threshold='auto', num_modes=100):
        """
        :param distance_threshold: clustering threshold
        :param num_modes: number of modes retained by modal truncation
        """
        self.distance_threshold = distance_threshold
        self.num_modes = num_modes

    def reduce(self, m_vector, m_gs, spring_Y, d_dashpot, d_drag, node_type):
        st_global = time()    
        n = len(m_vector)
        
        # 1. Identify node roles
        ctrl_indices = np.where(node_type == 3)[0]
        normal_indices = np.where(node_type != 3)[0]
        
        n_ctrl = len(ctrl_indices)
        n_normal = len(normal_indices)
        
        if n_normal <= 1:
            logging.warning("No normal nodes to reduce or system too small.")
            return np.eye(n), None, None, None

        # 2. Build physical matrices
        M_sp, D_sp, L_sp = build_physical_matrices_sparse(m_vector, 
                                                          m_gs, 
                                                          spring_Y, 
                                                          d_dashpot, 
                                                          d_drag)

        # =================================================================
        # 3. Extract force-receiving nodes (common logic moved earlier)
        # =================================================================
        u, v = m_gs[0], m_gs[1]
        connected_v = v[np.isin(u, ctrl_indices)]
        connected_u = u[np.isin(v, ctrl_indices)]
        target_nodes = np.unique(np.concatenate([connected_u, connected_v]))
        target_nodes = target_nodes[node_type[target_nodes] != 3]
        
        m_inputs = len(target_nodes) if len(target_nodes) > 0 else 1
        if len(target_nodes) == 0: target_nodes = np.array([0])

        # =================================================================
        # 4. Decide and compute the feature embedding Z (full-space Lyapunov vs. reduced-projection Lyapunov)
        # =================================================================
        if n < 2 * self.num_modes:
            # --- No modal decomposition: solve Lyapunov directly in the full physical space ---
            print(f"Node count ({n}) < 2*num_modes. Skipping eigsh, solving full-space Lyapunov.")
            
            # Construct the un-reduced system matrix A_full (2n x 2n)
            M_d = M_sp.diagonal()
            M_inv = sp.diags(1.0 / M_d, format='csr')
            Minv_L = (M_inv @ L_sp).toarray()
            Minv_D = (M_inv @ D_sp).toarray()
            
            A_full = np.block([
                [np.zeros((n, n)), np.eye(n)],
                [-Minv_L, -Minv_D]
            ])
            
            # Construct the un-reduced input matrix B_full_eff
            ones_n = np.ones((n, 1))
            sigma_D = (ones_n.T @ D_sp @ ones_n)[0, 0]
            ones_T_F = np.ones((1, m_inputs))
            
            F_full = np.zeros((n, m_inputs))
            for i, idx in enumerate(target_nodes):
                F_full[idx, i] = 1.0
                
            B_full_eff = np.block([
                [-(1.0 / sigma_D) * ones_n @ ones_T_F],
                [F_full]
            ])
            
            # Solve the full-dimensional Lyapunov equation
            try:
                P_tilde = solve_continuous_lyapunov(A_full - 1e-5 * np.eye(2 * n), -B_full_eff @ B_full_eff.T)
            except Exception as e:
                logging.error(f"Full-space Lyapunov failed: {e}")
                import pdb; pdb.set_trace()
                
            # Full-dimensional Gramian correction and embedding-space computation
            nu_full_T = np.block([[ones_n.T @ D_sp, ones_n.T @ M_sp]])
            beta_a = - (nu_full_T @ P_tilde @ nu_full_T.T)[0, 0] / (sigma_D ** 2)
            Pi_full = np.block([
                [ones_n @ ones_n.T, np.zeros((n, n))],
                [np.zeros((n, n)), np.zeros((n, n))]
            ])
            P_gramian = P_tilde + beta_a * Pi_full
            
            P11 = P_gramian[:n, :n]
            w, v_e = np.linalg.eigh(P11)
            w[w < 0] = 0
            C = v_e @ np.diag(np.sqrt(w))
            Z = C  # In the full space this is equivalent to Phi=I_n, so Z = I * C = C

        else:
            # --- Original logic: perform modal decomposition and solve Lyapunov in the low-dimensional space ---
            k_modes = min(self.num_modes, n - 2) 
            eigenvalues, Phi = eigsh(L_sp, k=k_modes, M=M_sp, sigma=-1e-4, which='LM')
            
            for i in range(k_modes):
                norm = np.sqrt(Phi[:, i].T @ M_sp @ Phi[:, i])
                if norm > 1e-12: Phi[:, i] /= norm

            F_r = Phi[target_nodes, :].T  
            ones_n = np.ones((n, 1))
            sigma_D = (ones_n.T @ D_sp @ ones_n)[0, 0]
            ones_T_F = np.ones((1, m_inputs)) 

            B_r_eff = np.block([
                [-(1.0 / sigma_D) * (Phi.T @ ones_n) @ ones_T_F],
                [F_r]
            ])

            M_r = Phi.T @ M_sp @ Phi
            D_r = Phi.T @ D_sp @ Phi
            L_r = np.diag(eigenvalues)
            
            A_r = np.block([
                [np.zeros((k_modes, k_modes)), np.eye(k_modes)],
                [-np.linalg.inv(M_r) @ L_r, -np.linalg.inv(M_r) @ D_r]
            ])
            
            try:
                P_r_tilde = solve_continuous_lyapunov(A_r - 1e-5 * np.eye(2 * k_modes), -B_r_eff @ B_r_eff.T)
            except:
                import pdb; pdb.set_trace()
                
            nu_r_T = np.block([[ones_n.T @ D_sp @ Phi, ones_n.T @ M_sp @ Phi]])
            beta_a = - (nu_r_T @ P_r_tilde @ nu_r_T.T)[0, 0] / (sigma_D ** 2)
            Pi_r = np.block([
                [Phi.T @ ones_n @ ones_n.T @ Phi, np.zeros((k_modes, k_modes))],
                [np.zeros((k_modes, k_modes)), np.zeros((k_modes, k_modes))]
            ])
            P_r_gramian = P_r_tilde + beta_a * Pi_r

            P_r11 = P_r_gramian[:k_modes, :k_modes]
            w, v_e = np.linalg.eigh(P_r11)
            w[w < 0] = 0
            C = v_e @ np.diag(np.sqrt(w))
            Z = Phi @ C

        # =================================================================
        # 5. Restricted clustering logic: cluster only normal_indices
        # =================================================================
        Z_normal = Z[normal_indices, :]
        normal_types = node_type[normal_indices]
        n_nodes = len(normal_indices)
        
        global_to_local = np.full(n, -1, dtype=np.int32)
        global_to_local[normal_indices] = np.arange(n_nodes)
        
        u, v = m_gs[0], m_gs[1]
        u_local = global_to_local[u]
        v_local = global_to_local[v]
        
        valid_mask = (u_local >= 0) & (v_local >= 0) & (u_local != v_local)
        u_local = u_local[valid_mask]
        v_local = v_local[valid_mask]
        
        cluster_labels = self._constrained_hierarchical_clustering_vectorized(
            Z_normal, u_local, v_local, n_nodes, normal_types
        )

        # =================================================================
        # 6. Build the N x N projection matrix P
        # =================================================================
        P = np.zeros((n, n))
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            in_cluster_mask = (cluster_labels == label)
            cluster_global_indices = normal_indices[in_cluster_mask]
            
            Z_cluster = Z[cluster_global_indices]
            centroid = np.mean(Z_cluster, axis=0)
            
            dists = np.linalg.norm(Z_cluster - centroid, axis=1)
            j = cluster_global_indices[np.argmin(dists)]
            P[cluster_global_indices, j] = 1.0

        for ctrl_idx in ctrl_indices:
            P[ctrl_idx, ctrl_idx] = 1.0

        return P

    def _constrained_hierarchical_clustering_vectorized(self, Z_normal, edge_src, edge_dst, n_nodes, node_types=None):
        """
        Perform constrained hierarchical clustering: only spatially connected (edge-connected) nodes with the same type may merge
        Implemented entirely with matrix operations to avoid for loops
        
        Args:
            Z_normal: embedding coordinates of normal nodes [n_normal, k]
            edge_src: local indices of the source nodes of edges [n_edges]
            edge_dst: local indices of the target nodes of edges [n_edges]
            n_nodes: number of normal nodes
            node_types: type array for each normal node (constrains merging to nodes of the same type)
        
        Returns:
            cluster_labels: cluster label for each node
        """
        Z_src = Z_normal[edge_src]
        Z_dst = Z_normal[edge_dst]
        edge_distances = np.linalg.norm(Z_src - Z_dst, axis=1)

        dist_threshold = np.percentile(edge_distances, 10)

        # 3. Build type-consistency constraints + physical connectivity constraints
        type_mask = (node_types[edge_src] == node_types[edge_dst])
        valid_u = edge_src[type_mask]
        valid_v = edge_dst[type_mask]
        
        # Build the connectivity matrix (only connected points of the same type can merge)
        connectivity = sp.coo_matrix(
            (np.ones(len(valid_u)), (valid_u, valid_v)),
            shape=(n_nodes, n_nodes)
        )

        # 4. Perform hierarchical clustering
        # n_clusters=None must be used together with distance_threshold
        # linkage='average' (average distance) prevents everything from collapsing into one cluster, unlike the previous connected-components method
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_threshold,
            connectivity=connectivity,
            linkage='average' 
        )
        
        labels = model.fit_predict(Z_normal)

        # --- Added: print the compression status of normal nodes ---
        n_clusters = len(np.unique(labels))
        reduction_pct = (1 - n_clusters / n_nodes) * 100
        print(f"\n[Clustering Detail]")
        print(f" - Normal Nodes: {n_nodes} -> {n_clusters}")
        print(f" - Reduction Ratio: {reduction_pct:.2f}%")
        print(f" - Adaptive Threshold: {dist_threshold:.6f}")

        return labels


class RandomReducer:
    def __init__(self, ratio=0.5, seed=None):
        """
        Random reducer that randomly merges normal nodes
        
        :param ratio: fraction of normal nodes finally retained, default 0.5
        :param seed: random seed for reproducibility
        """
        self.ratio = ratio
        self.seed = seed

    def reduce(self, m_vector, m_gs, spring_Y, d_dashpot, d_drag, node_type):
        """
        Perform random dimensionality reduction
        
        Args:
            m_vector: node mass vector [n]
            m_gs: edge connectivity [2, n_edges]
            spring_Y: spring stiffness coefficients [n_edges]
            d_dashpot: damping coefficient
            d_drag: drag coefficient
            node_type: node type array [n], type==3 means controller
        
        Returns:
            P: projection matrix [n, n]
            M_hat: reduced mass matrix [n, n]
            D_hat: reduced damping matrix [n, n]
            L_hat: reduced Laplacian matrix [n, n]
        """
        st_global = time()
        n = len(m_vector)
        
        # 1. Identify node roles
        # node_type == 3 means controller
        ctrl_indices = np.where(node_type == 3)[0]
        normal_indices = np.where(node_type != 3)[0]
        
        n_ctrl = len(ctrl_indices)
        n_normal = len(normal_indices)
        
        if n_normal <= 1:
            logging.warning("No normal nodes to reduce or system too small.")
            return np.eye(n), None, None, None

        # 2. Build physical matrices (using the provided sparse builder)
        M_sp, D_sp, L_sp = build_physical_matrices_sparse(m_vector, 
                                                          m_gs, 
                                                          spring_Y, 
                                                          d_dashpot, 
                                                          d_drag)

        # 3. Randomly merge normal nodes
        # Set the random seed to ensure reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Compute the number of target nodes to retain
        target_count = max(1, int(n_normal * self.ratio))
        
        # Randomly select target node indices among normal nodes
        target_indices = rng.choice(n_normal, size=target_count, replace=False)
        target_indices.sort()
        
        # Create the mapping array: each normal node maps to one target node
        # -1 means this node is itself a target node
        mapping = np.full(n_normal, -1, dtype=np.int64)
        
        # Randomly assign non-target nodes to target nodes
        non_target_indices = np.setdiff1d(np.arange(n_normal), target_indices)
        if len(non_target_indices) > 0:
            # Randomly assign to the various target nodes
            assignments = rng.randint(0, target_count, size=len(non_target_indices))
            for i, nt_idx in enumerate(non_target_indices):
                mapping[nt_idx] = target_indices[assignments[i]]
        
        # 4. Build the projection matrix P [n x n]
        P = np.eye(n)
        
        # Set the mapping for non-target nodes: merge these nodes into their corresponding target nodes
        for i, normal_idx in enumerate(normal_indices):
            if mapping[i] >= 0:
                # This node must be merged into the target node
                target_normal_idx = normal_indices[mapping[i]]
                P[normal_idx, target_normal_idx] = 1.0
                P[normal_idx, normal_idx] = 0.0

        # import pdb; pdb.set_trace()

        # # 5. Compute the reduced physical matrices
        # M_hat = P.T @ M_sp @ P
        # D_hat = P.T @ D_sp @ P
        # L_hat = P.T @ L_sp @ P

        elapsed = time() - st_global
        actual_reduced = n - len(non_target_indices) if len(non_target_indices) > 0 else n
        logging.info(f"Random Reduction done: {n} nodes ({n_ctrl} ctrl) -> {actual_reduced} nodes. "
                     f"Normal nodes compressed from {n_normal} to {target_count}. "
                     f"Time: {elapsed:.3f}s")

        return P
