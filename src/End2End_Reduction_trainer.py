import torch
from qqtt.utils import logger, cfg
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from End2End_EvoSpring_Reduction import End2EndReduction_EvoSpring
from End2End_Reduction_Dataset import End2EndReductionDataset
import math
import numpy as np
import os
from Reducer import FastAdaptiveNetworkReducer, RandomReducer, build_physical_matrices_sparse
import warp as wp
from time import time
import pickle
import csv
import logging
from datetime import datetime

# Disable all warp-related loggers
logging.getLogger("warp").setLevel(logging.ERROR)


class E2EReductionTrainer:
    def __init__(self, args, device="cuda:0",):
        self.args = args
        cfg.device = device

        self.mdata = self._create_dataset_offline(mode='train')
        # Per-object initial stiffness from configs/initial_training_settings.json
        # (applied after the dataset loads its yaml, which would otherwise overwrite it)
        if getattr(self.args, 'init_spring_Y', None) is not None:
            cfg.init_spring_Y = self.args.init_spring_Y

        self._create_model(cfg.init_spring_Y,
                            cfg.drag_damping,
                            cfg.dashpot_damping,
                            cfg.collide_elas,
                            cfg.collide_fric,
                            cfg.collide_object_elas,
                            cfg.collide_object_fric)
        
        # Initialize the Warp optimizer (only in training mode with collision_learn enabled)
        # Update without warp, only by NN for multilayer design
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr * min(np.sqrt(cfg.train_frame), 1), 
                                          betas=(0.9, 0.99))

        self.epochs_per_stage = 20 # 20
        self.iter_per_epoch = 10
        
        # Variables related to staged training
        self.num_stages = self.args.multi_mesh_layer  # number of stages equals the number of layers
        self.current_stage = 0  # current training stage
        
        # Create the save folder based on the program run time
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_base_dir = os.path.join(self.args.dump_dir, self.run_timestamp)
        os.makedirs(self.save_base_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.save_base_dir}")
        
        # Create all required subdirectories under the timestamp folder
        for subdir in ['ckpts', 'log', 'test_RMSE', 'spring_mech_info', 'trajectories', 'exported_meshes', 'visualization']:
            dir = os.path.join(self.save_base_dir, subdir)
            os.makedirs(dir, exist_ok=True)
        
        self.writer = SummaryWriter(os.path.join(self.save_base_dir, 'log'))

        # Per-stage (per-layer) training timing, written to CSV after every stage
        self.stage_time_csv_path = os.path.join(self.save_base_dir, 'stage_training_times.csv')
        self.stage_timing_records = []
        self._init_stage_timing_csv()

        self.total_update = 0
        
        # Track the best results independently for each stage
        self.stage_best_losses = []  # list of the best loss for each stage
        self.stage_best_mech_info = []  # list of the best mech_info for each stage
        self.stage_best_epochs = []  # best epoch for each stage
        self.stage_best_iterations = []  # best iteration for each stage
        
        # Initialize the per-stage best results
        for _ in range(self.num_stages):
            self.stage_best_losses.append(float('inf'))
            self.stage_best_mech_info.append(None)
            self.stage_best_epochs.append(None)
            self.stage_best_iterations.append(None)
        
        # Global best results (used as reference during downsampling)
        self.global_best_loss = float('inf')
        self.global_best_mech_info = []
        self.global_best_epoch = None
        self.global_best_iteration = None

        self.mlvl_simulators = []
        self.mlvl_collide_optimizer = []
        
        if args.reduction == 'learnable':
            self.use_reduction_downsample = False
        elif args.reduction == 'reduction_order':
            self.use_reduction_downsample = True
            self.reducer = FastAdaptiveNetworkReducer(num_modes=500)
        elif args.reduction == 'random':
            self.use_reduction_downsample = True
            self.reducer = RandomReducer(0.6)

    def create_simulator(self, stage):
        # create simulator here
        vertice = self.m_vertices[stage]
        masses = self.m_masses[stage]
        
        spring_graph = self.m_gs[stage].int().T.contiguous()
        num_edge = len(spring_graph)
        spring_graph = spring_graph[:num_edge // 2]
        node_type = self.m_node_type[stage]
        
        rest_lengths = torch.norm((vertice[spring_graph[:, 0]]
                                   -vertice[spring_graph[:, 1]]), dim = 1)
  
        # downsample for gt points
        gt_object_points = self.m_gt_object_points[stage]
        gt_object_visibilities = self.m_gt_object_visibilities[stage]
        gt_object_motions_valid = self.m_gt_object_motions_valid[stage]

        # create simulator
        simulator = self.mdata.create_spring_mass_sim(vertice, 
                                                    spring_graph, 
                                                    rest_lengths, 
                                                    masses,
                                                    node_type,
                                                    gt_object_points, 
                                                    gt_object_visibilities, 
                                                    gt_object_motions_valid)

        # Initialize the Warp optimizer (only in training mode with collision_learn enabled)
        if cfg.collision_learn:
            # Ensure requires_grad=True
            # Collision parameters
            torch_collide_elas = wp.to_torch(simulator.wp_collide_elas)
            torch_collide_fric = wp.to_torch(simulator.wp_collide_fric)
            torch_collide_object_elas = wp.to_torch(simulator.wp_collide_object_elas)
            torch_collide_object_fric = wp.to_torch(simulator.wp_collide_object_fric)

            # Add all differentiable parameters to the list
            warp_params_list = [
                torch_collide_elas,
                torch_collide_fric,
                torch_collide_object_elas,
                torch_collide_object_fric,
            ]
        else:
            warp_params_list = []

        collide_optimizer = torch.optim.Adam(warp_params_list,
                                          lr=self.args.lr,
                                          betas=(0.9, 0.99))
        self.mlvl_simulators.append(simulator)
        self.mlvl_collide_optimizer.append(collide_optimizer)

    def downsample(self, stage_idx):
        # 1. Heuristic threshold setting: the later the stage, the larger the allowed difference for merging

        logger.info(f"Applying adaptive downsampling for Stage {stage_idx}...")
        
        node_mass = self.m_masses[-1].cpu().numpy()
        gs = self.m_gs[-1].cpu().numpy()

        gs = gs[:, :len(gs[0])//2]

        # Get the globally optimal mechanical parameters
        best_mech = self.stage_best_mech_info[stage_idx]
        spring_Y = np.exp(best_mech['log_spring_Y'].cpu().numpy())
        drag_damping = best_mech['drag_damping'].item()
        dashpot_damping = best_mech['dashpot_damping'].item()

        node_type = self.m_node_type[-1].cpu().numpy()

        # 2. Call the adaptive reducer to perform clustering
        # Internally, the reducer converts physical parameters into M, D, L matrices and performs Gram matrix clustering
        P_np = self.reducer.reduce(
            node_mass, gs, spring_Y, dashpot_damping, drag_damping, node_type
        )

        # Save the projection matrix P for feature aggregation during the neural network forward pass
        P_tensor = torch.tensor(P_np, device=cfg.device).float()
        self.m_proj.append(P_tensor)
      
    def export_multi_level_obj(self, save_dir="exported_meshes"):
        """
        Export the reduced graph structure (vertices and edges) of each layer as a standard .obj file.
        Vertex coordinates directly keep the original physical coordinates of the representative nodes (no centroid averaging).
        
        Change note: export to the exported_meshes directory under the timestamp folder
        """
        logger.info(f"Exporting multi-level topology to {save_dir}...")
        
        # Create the export directory - saved under the timestamp folder
        export_path = os.path.join(self.save_base_dir, save_dir)
        os.makedirs(export_path, exist_ok=True)
        
        # ==========================================
        # 1. Prepare the global original coordinates
        # ==========================================
        if isinstance(self.mdata.init_vertices, torch.Tensor):
            global_vertices = self.mdata.init_vertices.detach().cpu().numpy()
        else:
            global_vertices = self.mdata.init_vertices
            
        # ==========================================
        # 2. Export layer 0 (original high-resolution structure)
        # ==========================================
        current_edges = self.m_gs[0].detach().cpu().numpy()  # [2, num_edges]
        
        self._write_obj(
            filename=os.path.join(export_path, f"layer_0_nodes_{global_vertices.shape[0]}.obj"),
            vertices=global_vertices,
            edges=current_edges
        )
        
        # ==========================================
        # 3. Extract representative nodes layer by layer and export them
        # ==========================================
        # self.m_ids stores the original indices of each layer's nodes in the global graph
        # Note: m_ids[0] may be the full set; m_ids[1] is the result of the first downsampling
        
        for layer_idx in range(1, len(self.m_ids)):
            # Get the global node indices kept in the current layer (e.g. [3, 5, 9, 12, ...])
            current_node_ids = self.m_ids[layer_idx]
            current_points = global_vertices[current_node_ids]
            current_edges = self.m_gs[layer_idx].detach().cpu().numpy()

            # Export the current layer
            filename = os.path.join(export_path, f"layer_{layer_idx}_nodes_{current_points.shape[0]}.obj")
            self._write_obj(filename, current_points, current_edges)
            
        logger.info(f"Successfully exported {len(self.m_ids)} layers of .obj files to {export_path}")

    def _write_obj(self, filename, vertices, edges):
        """
        Internal helper: write vertices and edges in OBJ file format
        """
        with open(filename, 'w') as f:
            f.write(f"# Auto-generated hierarchical point cloud\n")
            f.write(f"# Vertices: {vertices.shape[0]}, Edges: {edges.shape[1]}\n\n")
            
            # 1. Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            f.write("\n")
            
            # 2. Write edges/line segments (Lines)
            if edges.shape[1] > 0:
                edges_sorted = np.sort(edges, axis=0)
                edges_unique = np.unique(edges_sorted, axis=1)
                
                for col in range(edges_unique.shape[1]):
                    # OBJ indices start at 1
                    u = edges_unique[0, col] + 1  
                    v = edges_unique[1, col] + 1
                    f.write(f"l {u} {v}\n")
        
    def _create_model(self, init_spring_Y=None, init_drag_damping=None, init_dashpot_damping=None, init_collision_elas=None, init_collision_fric=None, init_collision_object_elas=None, init_collision_object_fric=None):
        self.object_case = self.args.object_case

        self.model = End2EndReduction_EvoSpring(
            pos_dim=self.args.space_dim,
            ld=self.args.hidden_dim,
            layer_num=self.args.multi_mesh_layer,
            pre_layer_num=self.args.pre_layer_num,
            bottom_layer_num=self.args.bottom_layer_num,
            mlp_hidden_layer=self.args.hidden_depth,
            MP_times=self.args.mp_time,
            enhance=self.args.enhance,
            agg_conv_pos=self.args.agg_conv_pos,
            default_spring_Y=init_spring_Y,
            default_drag_damping=init_drag_damping,
            default_dashpot_damping=init_dashpot_damping,
            default_collision_elas=init_collision_elas,
            default_collision_fric=init_collision_fric,
            default_collision_object_elas=init_collision_object_elas,
            default_collision_object_fric=init_collision_object_fric
        )

        self.model.to(self.args.local_rank)


    def _create_dataset_offline(self, mode='train', stride=1):
        if mode == 'train':
            mdata = End2EndReductionDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        object_case=self.args.object_case,
                                        args=self.args, device=cfg.device)
        else:
            mdata = End2EndReductionDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        mode=mode,
                                        object_case=self.args.object_case,  
                                        args=self.args, device=cfg.device)
        return mdata

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

    def save_traj(
        self, mlvl_mech_info=None, save_path=None, compute_loss=False, gt_indices=None, stage_idx=None,
        mlvl_masses=None, mlvl_edges=None
    ):
        """
        Save trajectory data, optionally saving ground-truth downsampling indices (for multi-stage evaluation)
        
        Args:
            mlvl_mech_info: list of mechanical parameters
            save_path: save path
            compute_loss: whether to compute loss
            gt_indices: list of ground-truth downsampling indices (one index list per stage)
            stage_idx: current stage index (used to extract the corresponding stage's indices from gt_indices)
            mlvl_masses: list of particle masses for each level (optional)
            mlvl_edges: list of connectivity/edge topology for each level (optional)
        """
        # ====================================================================
        # [Core change]: save trajectory results for all levels
        # ====================================================================
        logger.info("Initializing all level simulators for trajectory rollout...")
        
        # Save results for all levels
        all_level_vertices = []
        all_level_velocities = []
        
        for level_idx, simulator in enumerate(self.mlvl_simulators):
            logger.info(f"Processing Level {level_idx}...")
            
            # ====================================================================
            # 2. Assign the predicted mechanical parameters (mech_info) to the simulator
            # ====================================================================
            if mlvl_mech_info is not None and len(mlvl_mech_info) > level_idx:
                logger.info(f"Setting predicted mechanical properties to simulator (Level {level_idx})")
                level_mech_info = mlvl_mech_info[level_idx]

                if isinstance(level_mech_info, dict):
                    if 'log_spring_Y' in level_mech_info:
                        wp_predicted_spring_Y = wp.from_torch(level_mech_info['log_spring_Y'].contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_spring_Y(wp_predicted_spring_Y)

                    if 'drag_damping' in level_mech_info:
                        wp_predicted_drag_damping = wp.from_torch(level_mech_info['drag_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_drag_damping(wp_predicted_drag_damping)

                    if 'dashpot_damping' in level_mech_info:
                        wp_predicted_dashpot_damping = wp.from_torch(level_mech_info['dashpot_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_dashpot_damping(wp_predicted_dashpot_damping)

                    if 'collision_elas' in level_mech_info:
                        wp_predicted_collision_elas = wp.from_torch(level_mech_info['collision_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_elas(wp_predicted_collision_elas)

                    if 'collision_fric' in level_mech_info:
                        wp_predicted_collision_fric = wp.from_torch(level_mech_info['collision_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_fric(wp_predicted_collision_fric)

                    if 'collision_object_elas' in level_mech_info:
                        wp_predicted_collision_object_elas = wp.from_torch(level_mech_info['collision_object_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_object_elas(wp_predicted_collision_object_elas)

                    if 'collision_object_fric' in level_mech_info:
                        wp_predicted_collision_object_fric = wp.from_torch(level_mech_info['collision_object_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_object_fric(wp_predicted_collision_object_fric)
                else:
                    logger.warning(f"level_mech_info for level {level_idx} has unexpected format, skipping property update")
            else:
                logger.warning(f"No mech_info available for level {level_idx}")

            # ====================================================================
            # 3. Run the forward simulation and record trajectories
            # ====================================================================
            frame_len = cfg.train_frame + cfg.test_frame
            simulator.set_init_state(
                simulator.wp_init_vertices, simulator.wp_init_velocities
            )

            vertices = [wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu()]
            velocities = [wp.to_torch(simulator.wp_states[0].wp_v, requires_grad=False).cpu()]

            frame_losses, chamfer_losses, track_losses = [0], [0], [0]

            for i in tqdm(range(1, frame_len), desc=f"Saving trajectory Level {level_idx}", leave=False):
                if cfg.data_type == "real":
                    simulator.set_controller_target(i, pure_inference=False)
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(simulator.forward_graph)
                else:
                    simulator.step()

                if compute_loss:
                    if cfg.data_type == "real":
                        simulator.calculate_loss()
                        chamfer_losses.append(wp.to_torch(simulator.chamfer_loss, requires_grad=False).item())
                        track_losses.append(wp.to_torch(simulator.track_loss, requires_grad=False).item())
                    else:
                        simulator.calculate_simple_loss()
                        chamfer_losses.append(0.0)
                        track_losses.append(0.0)

                    frame_losses.append(wp.to_torch(simulator.loss, requires_grad=False).item())
                    simulator.clear_loss()

                x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
                v = wp.to_torch(simulator.wp_states[-1].wp_v, requires_grad=False)
                vertices.append(x.cpu())
                velocities.append(v.cpu())
                simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v)

            vertices = torch.stack(vertices, dim=0)
            velocities = torch.stack(velocities, dim=0)
            
            all_level_vertices.append(vertices.cpu().numpy())
            all_level_velocities.append(velocities.cpu().numpy())
        
        # ====================================================================
        # 4. Package and save the results (.pkl)
        # ====================================================================
        logger.info(f"Save the trajectory to {save_path}")

        # Build save_data, containing trajectories and physical attributes for all levels
        save_data = {
            'vertices': all_level_vertices,       # trajectory list for all levels
            'velocities': all_level_velocities,   # velocity list for all levels
        }
        
        # Save node type information for each level
        save_data['node_types'] = []
        for level_idx in range(len(self.mlvl_simulators)):
            if hasattr(self, 'm_node_type') and level_idx < len(self.m_node_type):
                level_node_type = self.m_node_type[level_idx]
                if isinstance(level_node_type, torch.Tensor):
                    save_data['node_types'].append(level_node_type.cpu().numpy())
                else:
                    save_data['node_types'].append(level_node_type)
            else:
                # If there is no node_type information, use the default value (all 0)
                num_nodes = all_level_vertices[level_idx].shape[1]
                save_data['node_types'].append(np.zeros(num_nodes, dtype=np.int32))
        logger.info(f"Saved node_types for {len(save_data['node_types'])} levels")
        
        # Save particle mass information for each level
        if mlvl_masses is not None:
            save_data['masses'] = []
            for level_idx, level_masses in enumerate(mlvl_masses):
                if isinstance(level_masses, torch.Tensor):
                    save_data['masses'].append(level_masses.detach().cpu().numpy())
                else:
                    save_data['masses'].append(level_masses)
            logger.info(f"Saved masses for {len(save_data['masses'])} levels")
        
        # Save connectivity (edge topology) for each level
        
        if mlvl_edges is not None:
            save_data['edges'] = []
            for level_idx, level_edges in enumerate(mlvl_edges):
                if isinstance(level_edges, torch.Tensor):
                    save_data['edges'].append(level_edges.cpu().numpy())
                else:
                    save_data['edges'].append(level_edges)
            logger.info(f"Saved edges for {len(save_data['edges'])} levels")
        
        # If gt_indices and stage_idx are provided, save the ground-truth indices for the corresponding stage
        if gt_indices is not None and stage_idx is not None:
            if stage_idx < len(gt_indices):
                save_data['gt_indices'] = gt_indices[stage_idx]
                logger.info(f"Saved GT indices for stage {stage_idx}: {len(save_data['gt_indices'])} nodes")
            else:
                logger.warning(f"stage_idx {stage_idx} out of range for gt_indices (len={len(gt_indices)})")
        
        # Also save the current stage_idx for identification during evaluation
        if stage_idx is not None:
            save_data['stage_idx'] = stage_idx

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Trajectory saved successfully with {len(all_level_vertices)} levels")
        if compute_loss:
            return frame_losses, chamfer_losses, track_losses, save_data

    def _preproc_multi_infos(self, mdata):
        # no contact, then share the graph between batches
        # only keep the first layer
        self.m_ids = [[i for i in range(len(mdata.init_vertices))]]
        m_gs_list = [np.concatenate([mdata.cells, mdata.cells[[1,0]]], axis = 1)]
        self.m_gs = [torch.tensor(g, dtype=torch.long).to(cfg.device) for g in m_gs_list]

        # Projection compression matrix
        self.m_proj = []
        self.m_vertices = [torch.tensor(mdata.init_vertices, dtype=torch.float32, device=cfg.device)]
        self.m_masses = [mdata.init_masses]
        self.m_node_type = [torch.tensor(mdata.node_type[0, :, 0], dtype=torch.long, device=cfg.device)]

        self.m_gt_object_points = [mdata.object_points.clone().to(cfg.device)]
        self.m_gt_object_visibilities = [mdata.object_visibilities.clone().to(cfg.device)]
        self.m_gt_object_motions_valid = [mdata.object_motions_valid.clone().to(cfg.device)]


    def _compute_abs_node_ids(self, stage: int):
        """Compute the absolute index in stage 0 (the first layer) for each node of the current stage.

        self.m_ids[l] stores the indices of layer l nodes within layer l-1 (relative mapping).
        To obtain absolute indices, all mappings from the current stage to stage 0 must be chained together.

        For example, the i-th node of stage=3:
            idx = self.m_ids[3][i]           # -> index in stage 2
            idx = self.m_ids[2][idx]         # -> index in stage 1
            idx = self.m_ids[1][idx]         # -> index in stage 0

        Args:
            stage: current stage number

        Returns:
            list[int]: absolute index in stage 0 for each node, length = number of nodes in the current stage
        """
        # Start from the local indices of the current layer (0, 1, 2, ..., N-1)
        abs_ids = torch.arange(len(self.m_ids[stage]), dtype=torch.long)

        # Map upward layer by layer: stage -> stage-1 -> ... -> stage 0
        for l in range(stage, 0, -1):
            mapping = torch.tensor(self.m_ids[l], dtype=torch.long)
            abs_ids = mapping[abs_ids]

        return abs_ids.tolist()


    def generate_data_point_sequence(self, simulator, update_frame_num, enable_backward=False):
        vertices_sequence = []
        velocities_sequence = []

        # Store gradient sequences for all differentiable parameters
        grad_sequences = {
            'log_spring_Y': [],
            'collision_elas': [],
            'collision_fric': [],
            'collision_object_elas': [],
            'collision_object_fric': [],
            'drag_damping': [],
            'dashpot_damping': []
        }

        # [Change] Define the Warp-side loss accumulator
        losses = []
        chamfer_losses = []
        track_losses = []

        # Update cfg.device to a Warp-compatible string or directly obtain a Warp device object
        simulator.set_init_state(
            simulator.wp_init_vertices, 
            simulator.wp_init_velocities
        )
        
        vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_x).clone(), wp.to_torch(simulator.wp_states[0].wp_control_x).clone()]))
        velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_v).clone(), wp.to_torch(simulator.wp_states[0].wp_control_v).clone()]))

        valid_frames = 0
        
        for frame_idx in tqdm(range(1, update_frame_num), leave=False):
            simulator.set_controller_target(frame_idx)

            # Record data
            vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_x).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_x).clone()]))
            velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_v).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_v).clone()]))

            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            # Compute loss
            if cfg.use_graph:
                wp.capture_launch(simulator.graph)
            else:
                if cfg.data_type == "real":
                    with simulator.tape:
                        simulator.step()
                        simulator.calculate_loss()
                    simulator.tape.backward(simulator.loss)
                else:
                    with simulator.tape:
                        simulator.step()
                        simulator.calculate_simple_loss()
                    simulator.tape.backward(simulator.loss)

            valid_frames += 1

            # Extract gradients of all differentiable parameters
            grad_dict = {}

            # spring_Y gradient
            if simulator.wp_spring_Y.grad is not None:
                grad_dict['log_spring_Y'] = wp.to_torch(simulator.wp_spring_Y.grad).clone()
            else:
                grad_dict['log_spring_Y'] = torch.zeros_like(wp.to_torch(simulator.wp_spring_Y))

            # Collision parameter gradients
            if cfg.collision_learn:
                # Damping parameter gradients
                # Check whether the simulator has Warp array versions of the damping parameters
                if hasattr(simulator, 'wp_drag_damping') and simulator.wp_drag_damping.grad is not None:
                    grad_dict['drag_damping'] = wp.to_torch(simulator.wp_drag_damping.grad).clone()
                elif hasattr(simulator, 'wp_drag_damping'):
                    grad_dict['drag_damping'] = torch.zeros_like(wp.to_torch(simulator.wp_drag_damping))
                else:
                    # If there is no Warp array version, create a scalar zero gradient
                    grad_dict['drag_damping'] = torch.zeros(1, dtype=torch.float32, device=cfg.device)

                if hasattr(simulator, 'wp_dashpot_damping') and simulator.wp_dashpot_damping.grad is not None:
                    grad_dict['dashpot_damping'] = wp.to_torch(simulator.wp_dashpot_damping.grad).clone()
                elif hasattr(simulator, 'wp_dashpot_damping'):
                    grad_dict['dashpot_damping'] = torch.zeros_like(wp.to_torch(simulator.wp_dashpot_damping))
                else:
                    # If there is no Warp array version, create a scalar zero gradient
                    grad_dict['dashpot_damping'] = torch.zeros(1, dtype=torch.float32, device=cfg.device)


            # Extract chamfer_loss and track_loss (only for real data)
            if cfg.data_type == "real":
                chamfer_loss = wp.to_torch(simulator.chamfer_loss, requires_grad=False)
                track_loss = wp.to_torch(simulator.track_loss, requires_grad=False)
            else:
                chamfer_loss = torch.tensor(0.0)
                track_loss = torch.tensor(0.0)

            loss = wp.to_torch(simulator.loss, requires_grad=False)

            # Check whether gradients contain NaN
            has_nan = False

            for param_name, grad in grad_dict.items():
                if torch.isnan(grad).any():
                    print(f'Gradient of {param_name} turns to nan in step {frame_idx}')
                    has_nan = True

            if has_nan:
                break
            
            losses.append(loss.item())
            chamfer_losses.append(chamfer_loss.item())
            track_losses.append(track_loss.item())

            # Save all gradients
            for param_name, grad in grad_dict.items():
                grad_sequences[param_name].append(grad)
    
            if cfg.use_graph:
                # Only need to clear the gradient, the tape is created in the graph
                simulator.tape.zero()
            else:
                # Need to reset the compute graph and clear the gradient
                simulator.tape.reset()

            simulator.clear_loss()

            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v
            )

        # Convert the output tensors
        vertices_tensor = torch.stack(vertices_sequence, dim=0)
        velocities_tensor = torch.stack(velocities_sequence, dim=0)
        node_mass = torch.cat([wp.to_torch(simulator.wp_masses).clone(), torch.zeros(simulator.num_control_points, device=vertices_tensor.device)])
        spring_Y = wp.to_torch(simulator.wp_spring_Y).clone()

        spring_rest_length = wp.to_torch(simulator.wp_rest_lengths).clone()
        spring_dashpot_damping = simulator.wp_dashpot_damping
        drag_damping = simulator.wp_drag_damping

        if enable_backward:
            # Compute the average gradient of all parameters
            grad_avg_dict = {}

            for param_name, grad_list in grad_sequences.items():
                if len(grad_list) == 0:
                    continue
                else:
                    grad_avg_dict[param_name] = torch.sum(torch.stack(grad_list, dim=0), dim=0)

            # Return the average gradient dictionary
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_rest_length, spring_dashpot_damping, drag_damping, losses, chamfer_losses, track_losses, grad_avg_dict, valid_frames
        else:
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_rest_length, spring_dashpot_damping, drag_damping, losses, chamfer_losses, track_losses


    # ====================================================================
    # Per-stage (per-layer) timing helpers
    # ====================================================================
    STAGE_TIMING_FIELDS = [
        'stage', 'num_nodes', 'num_springs', 'epochs', 'iterations_per_epoch',
        'start_time', 'end_time', 'duration_sec', 'duration_min',
        'avg_epoch_sec', 'best_loss', 'best_epoch', 'best_iteration',
    ]

    def _init_stage_timing_csv(self):
        """Create the CSV file with a header as soon as the run starts."""
        with open(self.stage_time_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.STAGE_TIMING_FIELDS)
            writer.writeheader()

    def _write_stage_timing_csv(self):
        """(Re)write the full per-stage timing table (called after each stage)."""
        with open(self.stage_time_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.STAGE_TIMING_FIELDS)
            writer.writeheader()
            for record in self.stage_timing_records:
                writer.writerow(record)

    # ====================================================================
    # Train function: staged training
    # Each stage supervises the simulator of only one layer:
    # - Stage 0: supervise only the output of the top layer (no downsampling)
    # - Stage 1: supervise only the result after one downsampling
    # - and so on
    # Train each stage for epochs_per_stage epochs
    # The best results of each layer are saved independently and unaffected by later stages
    # ====================================================================
    def train(self):
        # Initialize initial graph structure info before loop
        self._preproc_multi_infos(self.mdata)
        
        # Create the overall progress bar
        total_epochs = self.num_stages * self.epochs_per_stage
        self.pbar = tqdm(total=total_epochs, unit="epoch")
        
        logger.info(f"Starting multi-stage training with {self.num_stages} stages, {self.epochs_per_stage} epochs per stage")
        
        # Staged training
        for stage in range(self.num_stages):
            # DEBUG(CXY)
            if stage >= 4:
                break

            self.current_stage = stage
            stage_start_time = time()
            stage_start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            epoch_durations = []
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Stage {stage} (supervising level {stage})")
            logger.info(f"{'='*60}\n")
            
            # Note: downsampling is performed via pooling operations inside the model forward pass
            # Each stage corresponds to one layer of model output; no manual downsampling is needed
            # The model forward pass returns all layer outputs and pooling_losses

            # Train all epochs of the current stage
            for epoch in range(self.epochs_per_stage):
                global_epoch = stage * self.epochs_per_stage + epoch
                logger.info(f"--- Stage {stage}, Epoch {epoch+1}/{self.epochs_per_stage} (Global Epoch {global_epoch+1}/{total_epochs}) ---")
                
                # Run an epoch, supervising only the current layer
                epoch_start_time = time()
                self.run_epoch(epoch, mode='train', stage=stage)
                epoch_durations.append(time() - epoch_start_time)

                # Save the model parameters of the current epoch
                epoch_checkpoint = {
                    'epoch': epoch,
                    'stage': stage,
                    'global_epoch': global_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'stage_best_losses': self.stage_best_losses,
                    'stage_best_epochs': self.stage_best_epochs,
                    'stage_best_iterations': self.stage_best_iterations,
                    'global_best_loss': self.global_best_loss,
                }
                epoch_ckpt_filename = f'stage{stage}_epoch{epoch}_global{global_epoch}_model_checkpoint.pth'
                epoch_ckpt_path = os.path.join(self.save_base_dir, 'ckpts', epoch_ckpt_filename)
                torch.save(epoch_checkpoint, epoch_ckpt_path)
                logger.info(f"Saved model checkpoint for Stage {stage}, Epoch {epoch} to {epoch_ckpt_path}")
                
                # Update the progress bar
                self.pbar.update(1)

            # Record the wall-clock training time of this layer/stage
            stage_duration = time() - stage_start_time
            try:
                num_nodes = int(self.m_vertices[stage].shape[0])
            except Exception:
                num_nodes = None
            try:
                # m_gs stores each spring in both directions, so half the column count
                # matches what create_simulator() actually simulates.
                num_springs = int(self.m_gs[stage].shape[1]) // 2
            except Exception:
                num_springs = None
            record = {
                'stage': stage,
                'num_nodes': num_nodes,
                'num_springs': num_springs,
                'epochs': len(epoch_durations),
                'iterations_per_epoch': self.iter_per_epoch,
                'start_time': stage_start_wall,
                'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration_sec': round(stage_duration, 3),
                'duration_min': round(stage_duration / 60.0, 3),
                'avg_epoch_sec': round(float(np.mean(epoch_durations)), 3) if epoch_durations else None,
                'best_loss': self.stage_best_losses[stage],
                'best_epoch': self.stage_best_epochs[stage],
                'best_iteration': self.stage_best_iterations[stage],
            }
            self.stage_timing_records.append(record)
            self._write_stage_timing_csv()
            logger.info(
                f"Stage {stage} training time: {record['duration_sec']:.2f}s "
                f"({record['duration_min']:.2f} min); CSV saved to {self.stage_time_csv_path}"
            )

            # # Save the best result of the current stage
            self._save_stage_result(stage)

            # Merge the best results of all stages into global_best
            self._merge_stage_results()

            if self.use_reduction_downsample:
                self.downsample(stage)
            
            logger.info(f"Stage {stage} completed. Best loss: {self.stage_best_losses[stage]:.6f}")
        
        # After all stages finish training, export and visualize
        try:
            self.export_multi_level_obj()
        except Exception as e:
            logger.error(f"Failed to export OBJ models: {e}")
        
        self.pbar.close()
        logger.info("Multi-stage training completed!")
    
    def _create_stage_simulator(self, stage):
        """
        Create the simulator for the current stage (cumulatively create all layers without clearing previous simulators)
        Obtain GT data by layer-by-layer downsampling
        """
        logger.info(f"Creating simulator for stage {stage}...")
        
        # [Fix] No longer clear previous simulators; instead accumulate simulators for all layers
        # Clear only at stage 0, ensuring each stage's simulator is retained
        if stage == 0:
            self.mlvl_simulators = []
            self.mlvl_collide_optimizer = []
        else:
            # For stage > 0, clear only the current stage and later parts (if any)
            # Keep the simulators of previous stages
            if len(self.mlvl_simulators) > stage:
                self.mlvl_simulators = self.mlvl_simulators[:stage]
                self.mlvl_collide_optimizer = self.mlvl_collide_optimizer[:stage]
        
        # Save the original GT data (layer 0)
        original_gt_points = self.mdata.object_points  # [T, N_original, ...]
        original_gt_vis = self.mdata.object_visibilities
        original_gt_valid = self.mdata.object_motions_valid
        
        # Downsample GT data layer by layer
        # Note: GT data corresponds only to object points (node_type=0)
        # Therefore, the indices of object points must be filtered out from current_ids
        gt_object_points_list = [original_gt_points]
        gt_object_visibilities_list = [original_gt_vis]
        gt_object_motions_valid_list = [original_gt_valid]
        
        # Downsample GT data layer by layer according to down_ids
        for lvl in range(1, stage + 1):
            # down_ids[lvl] stores the indices of the current layer's nodes within the previous layer
            prev_gt_points = gt_object_points_list[-1]
            prev_gt_vis = gt_object_visibilities_list[-1]
            prev_gt_valid = gt_object_motions_valid_list[-1]
            
            # Get the current layer's node_type and filter out object points (node_type=0)
            current_node_type = self.m_node_type[lvl]
            object_mask = (current_node_type == 0)
            object_indices = torch.where(object_mask)[0]
            
            # Extract only the indices corresponding to object points from current_ids
            # current_ids is a list and must be converted to a tensor for indexing
            current_ids = torch.tensor(self.m_ids[lvl], dtype=torch.long, device=cfg.device)
            object_ids = current_ids[object_indices]
            
            # Extract the current layer's GT data from the previous layer's GT data (only for object points)
            current_gt_points = prev_gt_points[:, object_ids]
            current_gt_vis = prev_gt_vis[:, object_ids]
            current_gt_valid = prev_gt_valid[:, object_ids]
            
            gt_object_points_list.append(current_gt_points)
            gt_object_visibilities_list.append(current_gt_vis)
            gt_object_motions_valid_list.append(current_gt_valid)
            logger.info(f"Downsampled GT data for level {lvl}: {prev_gt_points.shape} -> {current_gt_points.shape}")
        
        # [Key change] Save the layer-by-layer downsampled GT data back to the instance variable lists
        # This way, run_epoch can correctly obtain the downsampled GT data when saving the best results
        self.m_gt_object_points = gt_object_points_list
        self.m_gt_object_visibilities = gt_object_visibilities_list
        self.m_gt_object_motions_valid = gt_object_motions_valid_list
        
        # Use the current stage's vertices, masses, and other parameters
        vertice = self.m_vertices[stage]
        masses = self.m_masses[stage]
        
        spring_graph = self.m_gs[stage].int().T.contiguous()
        num_edge = len(spring_graph)
        spring_graph = spring_graph[:num_edge // 2]

        node_type = self.m_node_type[stage]
        
        rest_lengths = torch.norm((vertice[spring_graph[:, 0]]
                                   -vertice[spring_graph[:, 1]]), dim=1)
        
        # Use the layer-by-layer downsampled GT data
        gt_object_points = gt_object_points_list[stage]
        gt_object_visibilities = gt_object_visibilities_list[stage]
        gt_object_motions_valid = gt_object_motions_valid_list[stage]
        
        simulator = self.mdata.create_spring_mass_sim(vertice, 
                                                    spring_graph, 
                                                    rest_lengths, 
                                                    masses,
                                                    node_type,
                                                    gt_object_points, 
                                                    gt_object_visibilities, 
                                                    gt_object_motions_valid)
        
        if cfg.collision_learn:
            torch_collide_elas = wp.to_torch(simulator.wp_collide_elas)
            torch_collide_fric = wp.to_torch(simulator.wp_collide_fric)
            torch_collide_object_elas = wp.to_torch(simulator.wp_collide_object_elas)
            torch_collide_object_fric = wp.to_torch(simulator.wp_collide_object_fric)
            print(f"collide_elas {torch_collide_elas}")
            warp_params_list = [
                torch_collide_elas,
                torch_collide_fric,
                torch_collide_object_elas,
                torch_collide_object_fric,
            ]
        else:
            warp_params_list = []
        
        collide_optimizer = torch.optim.Adam(warp_params_list,
                                          lr=self.args.lr,
                                          betas=(0.9, 0.99))
        
        self.mlvl_simulators.append(simulator)
        self.mlvl_collide_optimizer.append(collide_optimizer)
        logger.info(f"Simulator created for stage {stage} with {vertice.shape[0]} nodes")
    
    def _save_stage_result(self, stage):
        """
        Save the best result of the current stage
        Keep the design of previous stages and cumulatively save the best results of all stages
        Each stage now contains complete downsampling information
        
        Change notes:
        1. Results are saved in the save folder created with a timestamp
        2. When saving each stage, cumulatively save all results from stage 0 to the current stage
        3. Call _save_accumulated_results to save the accumulated results of all stages
        """
        if self.stage_best_mech_info[stage] is None:
            logger.warning(f"No best mech info for stage {stage}, skipping save")
            return
        
        # Save the best mech_info of the current stage (already contains complete information)
        stage_mech_info = self.stage_best_mech_info[stage]
        
        # Accumulate the design of previous stages (if any)
        accumulated_mech_info = []
        for prev_stage in range(stage + 1):
            if self.stage_best_mech_info[prev_stage] is not None:
                accumulated_mech_info.append(self.stage_best_mech_info[prev_stage])
            else:
                logger.warning(f"Stage {prev_stage} has no best mech info, skipping")
        
        # Save the accumulated mech_info using the stage_epoch_iter format
        best_epoch = self.stage_best_epochs[stage]
        best_iter = self.stage_best_iterations[stage]
        
        # Save under the timestamp folder
        best_mech_filename = f'stage{stage}_epoch{best_epoch}_iter{best_iter}_accumulated_best_mech_info.pth'
        best_mech_info_path = os.path.join(self.save_base_dir, 'spring_mech_info', best_mech_filename)
        
        # Save the accumulated mech_info; each stage now contains complete information:
        # - log_spring_Y, drag_damping, dashpot_damping (mechanical parameters; Young's modulus saved as log)
        # - collision_elas, collision_fric, collision_object_elas, collision_object_fric (collision parameters)
        # - vertices (downsampled node positions: object + interior + surface + controller points)
        # - edges (downsampled connectivity)
        # - node_type (downsampled node types)
        # - masses (downsampled node masses)
        # - gt_vertices, gt_visibility, gt_motions_valid (downsampled GT data)
        # - node_ids (node ID mapping)
        torch.save({'mech': accumulated_mech_info}, best_mech_info_path)
        logger.info(f"Saved accumulated mech info with complete stage data to {best_mech_info_path}")
        
        # Save the best trajectory of the current stage (using the accumulated design; filename contains stage_epoch_iter)
        # Also save the accumulated trajectory results of all stages
        best_traj_filename = f'stage{stage}_epoch{best_epoch}_iter{best_iter}_best_trajectory.pkl'
        best_traj_path = os.path.join(self.save_base_dir, 'trajectories', best_traj_filename)
        
        # Extract the mechanical parameter part for save_traj (save_traj only needs mechanical parameters for simulation)
        mlvl_mech_info_for_traj = []
        for mech_info in accumulated_mech_info:
            traj_mech = {
                'log_spring_Y': mech_info['log_spring_Y'],
                'drag_damping': mech_info['drag_damping'],
                'dashpot_damping': mech_info['dashpot_damping'],
                'collision_elas': mech_info['collision_elas'],
                'collision_fric': mech_info['collision_fric'],
                'collision_object_elas': mech_info['collision_object_elas'],
                'collision_object_fric': mech_info['collision_object_fric'],
            }
            mlvl_mech_info_for_traj.append(traj_mech)
        
        self.save_traj(mlvl_mech_info=mlvl_mech_info_for_traj, 
                       save_path=best_traj_path, 
                       compute_loss=True,
                       gt_indices=self.m_ids,
                       stage_idx=stage,
                       mlvl_masses=self.m_masses,
                       mlvl_edges=self.m_gs)
        
        # Call _save_accumulated_results to save the accumulated results of all stages (including all trajectories from stage 0 to the current stage)
        self._save_accumulated_results(stage, accumulated_mech_info)
        
        logger.info(f"Stage {stage} best result saved with {len(accumulated_mech_info)} levels (stages 0-{stage})")
        logger.info(f"  Mech info: {best_mech_info_path}")
        logger.info(f"  Trajectory: {best_traj_path}")
    
    def _save_accumulated_results(self, current_stage, accumulated_mech_info):
        """
        Save the accumulated results of all stages (from stage 0 to current_stage)
        
        This function is called at the end of each stage to save the accumulated results of the current stage and all previous stages
        
        Args:
            current_stage: current stage index
            accumulated_mech_info: accumulated list of mechanical parameters for all stages
        """
        logger.info(f"Saving accumulated results for stages 0-{current_stage}...")
        
        # Save the accumulated results of all stages (including all trajectories from stage 0 to the current stage)
        accumulated_traj_filename = f'stage{current_stage}_accumulated_all_stages_trajectory.pkl'
        accumulated_traj_path = os.path.join(self.save_base_dir, 'trajectories', accumulated_traj_filename)
        
        # Generate trajectories for each completed stage
        all_stages_traj_data = []
        for completed_stage in range(current_stage + 1):
            if self.stage_best_mech_info[completed_stage] is not None:
                stage_traj_mech = {
                    'log_spring_Y': self.stage_best_mech_info[completed_stage]['log_spring_Y'],
                    'drag_damping': self.stage_best_mech_info[completed_stage]['drag_damping'],
                    'dashpot_damping': self.stage_best_mech_info[completed_stage]['dashpot_damping'],
                    'collision_elas': self.stage_best_mech_info[completed_stage]['collision_elas'],
                    'collision_fric': self.stage_best_mech_info[completed_stage]['collision_fric'],
                    'collision_object_elas': self.stage_best_mech_info[completed_stage]['collision_object_elas'],
                    'collision_object_fric': self.stage_best_mech_info[completed_stage]['collision_object_fric'],
                }
                all_stages_traj_data.append({
                    'stage': completed_stage,
                    'mech_info': stage_traj_mech,
                    'loss': self.stage_best_losses[completed_stage],
                    'epoch': self.stage_best_epochs[completed_stage],
                    'iter': self.stage_best_iterations[completed_stage],
                })
        
        # Save the accumulated information of all stages
        accumulated_save_data = {
            'all_stages_traj_data': all_stages_traj_data,
            'accumulated_mech_info': accumulated_mech_info,
            'current_stage': current_stage,
            'timestamp': self.run_timestamp,
        }
        
        with open(accumulated_traj_path, "wb") as f:
            pickle.dump(accumulated_save_data, f)
        
        logger.info(f"Saved accumulated all stages trajectory to {accumulated_traj_path}")
    
    def _merge_stage_results(self):
        """
        Merge the best results of all stages into global_best
        Each stage now contains complete downsampled data
        
        Change notes:
        1. Results are saved in the save folder created with a timestamp
        2. Also save a copy to the original spring_mech_info directory for compatibility
        """
        logger.info("Merging stage best results into global best...")
        
        self.global_best_mech_info = []
        for stage in range(self.num_stages):
            if self.stage_best_mech_info[stage] is not None:
                self.global_best_mech_info.append(self.stage_best_mech_info[stage])
                logger.info(f"  Stage {stage}: loss={self.stage_best_losses[stage]:.6f}, "
                           f"epoch={self.stage_best_epochs[stage]}, iter={self.stage_best_iterations[stage]}")
            else:
                logger.warning(f"  Stage {stage}: no best result available")
        
        # Save the merged global best results (including complete information for all stages)
        if len(self.global_best_mech_info) > 0:
            # Timestamp folder path
            timestamp_mech_info_path = os.path.join(self.save_base_dir, 'spring_mech_info', 'global_best_mech_info.pth')
            
            # global_best_mech_info now contains complete information for each stage:
            # For each stage:
            #   - log_spring_Y, drag_damping, dashpot_damping (mechanical parameters; Young's modulus saved as log)
            #   - collision_elas, collision_fric, collision_object_elas, collision_object_fric (collision parameters)
            #   - vertices (downsampled node positions: object + interior + surface + controller points)
            #   - edges (downsampled connectivity)
            #   - node_type (downsampled node types)
            #   - masses (downsampled node masses)
            #   - gt_vertices, gt_visibility, gt_motions_valid (downsampled GT data)
            #   - node_ids (node ID mapping)
            save_data = {'mech': self.global_best_mech_info}
            torch.save(save_data, timestamp_mech_info_path)
            logger.info(f"Saved global_best_mech_info with complete stage data to {timestamp_mech_info_path}")
            
            # Save the global best trajectory (extract the mechanical parameter part for simulation)
            # Timestamp folder path
            timestamp_traj_path = os.path.join(self.save_base_dir, 'trajectories', 'global_best_trajectory.pkl')
            
            # Extract mechanical parameters for save_traj
            mlvl_mech_info_for_traj = []
            for mech_info in self.global_best_mech_info:
                traj_mech = {
                    'log_spring_Y': mech_info['log_spring_Y'],
                    'drag_damping': mech_info['drag_damping'],
                    'dashpot_damping': mech_info['dashpot_damping'],
                    'collision_elas': mech_info['collision_elas'],
                    'collision_fric': mech_info['collision_fric'],
                    'collision_object_elas': mech_info['collision_object_elas'],
                    'collision_object_fric': mech_info['collision_object_fric'],
                }
                mlvl_mech_info_for_traj.append(traj_mech)
            
            # Save the trajectory to the timestamp folder
            self.save_traj(mlvl_mech_info=mlvl_mech_info_for_traj, 
                           save_path=timestamp_traj_path, 
                           compute_loss=False,
                           gt_indices=self.m_ids,
                           mlvl_masses=self.m_masses,
                           mlvl_edges=self.m_gs)
            
            logger.info(f"Global best trajectory saved to {timestamp_traj_path}")
            
            # Save a summary file recording the accumulated results of all stages
            all_stages_summary_path = os.path.join(self.save_base_dir, 'trajectories', 'all_stages_summary.pkl')
            
            # Build the summary data for all stages
            all_stages_summary = {
                'timestamp': self.run_timestamp,
                'num_stages': self.num_stages,
                'stage_results': []
            }
            
            for stage in range(self.num_stages):
                if self.stage_best_mech_info[stage] is not None:
                    stage_summary = {
                        'stage': stage,
                        'loss': self.stage_best_losses[stage],
                        'epoch': self.stage_best_epochs[stage],
                        'iter': self.stage_best_iterations[stage],
                        'mech_info': {
                            'log_spring_Y': self.stage_best_mech_info[stage]['log_spring_Y'],
                            'drag_damping': self.stage_best_mech_info[stage]['drag_damping'],
                            'dashpot_damping': self.stage_best_mech_info[stage]['dashpot_damping'],
                            'collision_elas': self.stage_best_mech_info[stage]['collision_elas'],
                            'collision_fric': self.stage_best_mech_info[stage]['collision_fric'],
                            'collision_object_elas': self.stage_best_mech_info[stage]['collision_object_elas'],
                            'collision_object_fric': self.stage_best_mech_info[stage]['collision_object_fric'],
                        },
                        'vertices_shape': self.stage_best_mech_info[stage]['vertices'].shape,
                        'edges_shape': self.stage_best_mech_info[stage]['edges'].shape,
                    }
                    all_stages_summary['stage_results'].append(stage_summary)
            
            with open(all_stages_summary_path, "wb") as f:
                pickle.dump(all_stages_summary, f)
            
            logger.info(f"Saved all stages summary to {all_stages_summary_path}")
            
            # Print a summary of the data saved for each stage
            for stage, mech_info in enumerate(self.global_best_mech_info):
                logger.info(f"\n  Stage {stage} saved data summary:")
                logger.info(f"    - Vertices shape: {mech_info['vertices'].shape}")
                logger.info(f"    - Edges shape: {mech_info['edges'].shape}")
                logger.info(f"    - Node types: {mech_info['node_type'].shape}")
                logger.info(f"    - Masses shape: {mech_info['masses'].shape if isinstance(mech_info['masses'], torch.Tensor) else len(mech_info['masses'])}")
                logger.info(f"    - GT vertices shape: {mech_info['gt_vertices'].shape}")
                logger.info(f"    - Spring Y (log) shape: {mech_info['log_spring_Y'].shape}")
                logger.info(f"    - Drag damping: {mech_info['drag_damping'].item():.6f}")
                logger.info(f"    - Dashpot damping: {mech_info['dashpot_damping'].item():.6f}")
    
    def run_epoch(self, epoch, mode='train', stage=0):
        """
        Run one epoch, supervising only the layer of the specified stage
        
        Args:
            epoch: current epoch index
            mode: 'train' or 'val'
            stage: current training stage (only the layer of this stage is supervised)
        """
        # get node_pos, node_mass, rest_length , drag_damping, dashpot_damping in first frame from mdata
        train_node_pos = self.mdata.init_vertices
        train_node_type = torch.FloatTensor(self.mdata.node_type[0]).to(cfg.device)
        train_node_mass = self.mdata.init_masses[:, None]

        # normalize train node pos into [-1,1]
        pos_min = train_node_pos.min(dim=0, keepdim=True)[0]
        pos_max = train_node_pos.max(dim=0, keepdim=True)[0]
        normalized_train_node_pos = 2 * (train_node_pos - pos_min) / (pos_max - pos_min) - 1

        # apply NeRF-style positional encoding to node positions
        pos_encoded = self.positional_encoding(normalized_train_node_pos, num_freq_bands=10)
        
        node_in_feature = torch.cat([
            normalized_train_node_pos, pos_encoded, train_node_mass, train_node_type
        ], dim=1)

        if mode != 'train':
            self.model.eval()

        # Run the simulation
        if mode == 'train':
            for i in range(self.iter_per_epoch):
                self.optimizer.zero_grad()
                
                st = time()

                # 1. Run the model forward pass to obtain downsampling results and mechanical parameters
                # Get the object_radius parameter from the config (distance constraint controlling merges)
                object_radius = getattr(cfg, 'object_radius', None)
                
                mlvl_s_out, mlvl_drag_damping_out, mlvl_dashpot_damping_out, downsample_results, pooling_losses = self.model(
                    self.m_ids, 
                    self.m_gs, 
                    self.m_proj,
                    self.m_vertices,
                    self.m_masses,
                    self.m_node_type,
                    node_in_feature,
                    object_radius,
                    self.global_best_mech_info
                )
                
                # In staged training, only the pooling loss of the current stage is used
                # pooling_losses contains the loss of each downpooling layer; we only take the loss corresponding to the current stage
                if stage < len(pooling_losses):
                    stage_pooling_loss = pooling_losses[stage]
                else:
                    stage_pooling_loss = torch.tensor(0.0, device=cfg.device)
                
                # 2. Update the topology using the downsampling results (only on the first iteration)
                logger.info(f"Updating topology structure at epoch {epoch}, iter {i+1}")
                self.m_ids = downsample_results['down_ids'][:stage+1]
                self.m_proj = downsample_results['down_proj'][:stage] # downsample proj number is level - 1
                self.m_vertices = downsample_results['down_ps'][:stage+1]
                self.m_masses = downsample_results['down_mass'][:stage+1]
                self.m_node_type = downsample_results['down_type'][:stage+1]
                self.m_gs = downsample_results['down_gs'][:stage+1]

                if epoch == 0 and i==0:
                    # Recreate the simulator for the current stage
                    self._create_stage_simulator(stage)

                # 3. Process only the layer of the current stage
                level_info = {
                    'log_spring_Y': mlvl_s_out[stage],
                    'drag_damping': mlvl_drag_damping_out[stage],
                    'dashpot_damping': mlvl_dashpot_damping_out[stage],
                }
                
                # [Fix] Use the simulator corresponding to the current stage
                current_simulator = self.mlvl_simulators[stage]
                
                # Set simulator parameters and run the simulation
                log_new_spring_Y = torch.log(level_info['log_spring_Y'])
                wp_predicted_spring_Y = wp.from_torch(
                    log_new_spring_Y.contiguous(), dtype=wp.float32, requires_grad=True
                )

                wp_predicted_drag_damping = wp.from_torch(
                    level_info['drag_damping'], dtype=wp.float32, requires_grad=True
                )
                    
                wp_predicted_dashpot_damping = wp.from_torch(
                    level_info['dashpot_damping'], dtype=wp.float32, requires_grad=True
                )

                current_simulator.set_spring_Y(wp_predicted_spring_Y)
                current_simulator.set_drag_damping(wp_predicted_drag_damping)
                current_simulator.set_dashpot_damping(wp_predicted_dashpot_damping)
                
                # Run the simulation sequence
                pos, _, _, _, _, _, _, loss_val, chamfer_loss_val, track_loss_val, grad_avg_dict, valid_frames = \
                    self.generate_data_point_sequence(
                        current_simulator,
                        update_frame_num=cfg.train_frame,
                        enable_backward=True,
                    )

                # Get collision parameters
                wp_collide_elas = wp.to_torch(current_simulator.wp_collide_elas).clone()
                wp_collide_fric = wp.to_torch(current_simulator.wp_collide_fric).clone()
                wp_collide_object_elas = wp.to_torch(current_simulator.wp_collide_object_elas).clone()
                wp_collide_object_fric = wp.to_torch(current_simulator.wp_collide_object_fric).clone()
                
                # Build the mech_info for the current stage
                current_mech_info = {
                    'log_spring_Y': log_new_spring_Y.detach().clone(),
                    'drag_damping': level_info['drag_damping'].detach().clone(),
                    'dashpot_damping': level_info['dashpot_damping'].detach().clone(),
                    'collision_elas': wp_collide_elas.detach().clone(),
                    'collision_fric': wp_collide_fric.detach().clone(),
                    'collision_object_elas': wp_collide_object_elas.detach().clone(),
                    'collision_object_fric': wp_collide_object_fric.detach().clone(),
                }
                
                # Compute loss
                simulation_loss = np.sum(loss_val)
                pooling_loss_value = stage_pooling_loss.item() * 0.000 
                total_loss_val = simulation_loss # + pooling_loss_value

                print("Forward time is : {}".format(time() - st))
                st = time()
                
                self.total_update += 1

                # Print information
                # Get spring rest_length statistics
                spring_rest_lengths = wp.to_torch(current_simulator.wp_rest_lengths, requires_grad=False)
                rest_length_min = spring_rest_lengths.min().item()
                rest_length_max = spring_rest_lengths.max().item()
                rest_length_mean = spring_rest_lengths.mean().item()
                
                print(f"\n{'='*60}")
                print(f"Stage {self.current_stage}, Epoch {epoch}")
                print(f"Spring Y iteration {i+1}/10")
                print(f"spring_Y grad Average: {grad_avg_dict['log_spring_Y'].mean().item()}")
                print(f"Spring_Y Average: {log_new_spring_Y.mean().item():.6f}")
                print(f"drag_damping grad Average: {grad_avg_dict['drag_damping'].mean().item()}")
                print(f"Drag_Damping: {level_info['drag_damping'].item():.6f}")
                print(f"dashpot_damping grad Average: {grad_avg_dict['dashpot_damping'].mean().item()}")
                print(f"Dashpot_Damping: {level_info['dashpot_damping'].item():.6f}")
                print(f"Simulation Loss: {simulation_loss:.10f}")
                print(f"Pooling Loss: {pooling_loss_value:.10f}")
                print(f"Total Loss: {total_loss_val:.10f}")
                print(f"Valid frames: {valid_frames}")
                print(f"Spring Rest Length - Min: {rest_length_min:.6f}, Max: {rest_length_max:.6f}, Mean: {rest_length_mean:.6f}")
                print(f"{'='*60}\n")

                self.writer.add_scalar(f'Stage_{self.current_stage}/overall_Loss', total_loss_val, self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/pooling_Loss', pooling_loss_value, self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/simulation_Loss', simulation_loss, self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/Spring_Y_Average', log_new_spring_Y.mean().item(), self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/Drag_Damping', level_info['drag_damping'].item(), self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/Dashpot_Damping', level_info['dashpot_damping'].item(), self.total_update)
 
                # Backpropagation
                tensors_to_backward = [log_new_spring_Y, level_info['drag_damping'], level_info['dashpot_damping']]
                grads_to_backward = [
                    grad_avg_dict['log_spring_Y'],
                    grad_avg_dict['drag_damping'],
                    grad_avg_dict['dashpot_damping'],
                ]
                
                # Add pooling loss
                tensors_to_backward.append(stage_pooling_loss)
                grads_to_backward.append(torch.ones_like(stage_pooling_loss)*0.000)
                
                torch.autograd.backward(
                    tensors=tensors_to_backward,
                    grad_tensors=grads_to_backward
                )

                self.optimizer.step()

                # Check and save the best result of the current stage
                if total_loss_val < self.stage_best_losses[self.current_stage]:
                    prev_best = self.stage_best_losses[self.current_stage]
                    self.stage_best_losses[self.current_stage] = total_loss_val

                    num_edge = self.m_gs[self.current_stage].shape[1]
                    
                    # Build the complete stage information, including all downsampled data
                    complete_stage_info = {
                        # 1. Mechanical parameters (spring Young's modulus saved as log)
                        'log_spring_Y': log_new_spring_Y.detach().clone(),
                        'drag_damping': level_info['drag_damping'].detach().clone(),
                        'dashpot_damping': level_info['dashpot_damping'].detach().clone(),
                        
                        # 2. Collision parameters
                        'collision_elas': wp_collide_elas.detach().clone(),
                        'collision_fric': wp_collide_fric.detach().clone(),
                        'collision_object_elas': wp_collide_object_elas.detach().clone(),
                        'collision_object_fric': wp_collide_object_fric.detach().clone(),
                        
                        # 3. Downsampled node positions (including object_point, interior point, surface point, controller point)
                        'vertices': self.m_vertices[self.current_stage].detach().clone(),
                        
                        # 4. Downsampled connectivity (edge topology)
                        'edges': self.m_gs[self.current_stage].detach().clone()[:, :num_edge//2],
                        
                        # 5. Downsampled node types
                        'node_type': self.m_node_type[self.current_stage].detach().clone(),
                        
                        # 6. Downsampled node masses
                        'masses': self.m_masses[self.current_stage].clone() if isinstance(self.m_masses[self.current_stage], torch.Tensor) else torch.tensor(self.m_masses[self.current_stage], device=cfg.device),
                        
                        # 7. Downsampled GT data (gt_vertices and visibility)
                        'gt_vertices': self.m_gt_object_points[self.current_stage].detach().clone(),
                        'gt_visibility': self.m_gt_object_visibilities[self.current_stage].detach().clone(),
                        'gt_motions_valid': self.m_gt_object_motions_valid[self.current_stage].detach().clone(),
                        
                        # 8. Node ID mapping (downsampling indices)
                        'node_ids': self.m_ids[self.current_stage],
                        'abs_node_ids': self._compute_abs_node_ids(self.current_stage),
                    }
                    
                    self.stage_best_mech_info[self.current_stage] = complete_stage_info
                    self.stage_best_epochs[self.current_stage] = epoch
                    self.stage_best_iterations[self.current_stage] = i
                    
                    print(f"\n{'='*60}")
                    print(f"NEW STAGE {self.current_stage} BEST!")
                    print(f"Epoch: {epoch}, Iteration: {i+1}/10")
                    print(f"Loss: {total_loss_val:.10f}")
                    print(f"Previous best: {prev_best:.10f}")
                    print(f"{'='*60}\n")
                    
                    # Immediately save the current best mech_info; filename contains the stage_epoch_iter tag
                    # Save under the timestamp folder
                    best_mech_filename = f'stage{self.current_stage}_epoch{epoch}_iter{i}_best_mech_info.pth'
                    best_mech_info_path = os.path.join(self.save_base_dir, 'spring_mech_info', best_mech_filename)
                    torch.save({'mech': complete_stage_info}, best_mech_info_path)
                    logger.info(f"Saved best mech info for stage {self.current_stage} to {best_mech_info_path}")

            # Collision parameter optimization
            if cfg.collision_learn:
                print(f"\n{'='*60}")
                print("Starting collide parameters optimization...")
                print(f"{'='*60}\n")
                
                # [Fix] Optimize collision parameters for the current stage's simulator
                # Use self.current_stage to index and obtain the corresponding simulator
                simulator = self.mlvl_simulators[self.current_stage]
                collide_optimizer = self.mlvl_collide_optimizer[self.current_stage]
                
                for opt_iter in range(5):
                    iter_losses = []
                    
                    simulator.set_init_state(
                        simulator.wp_init_vertices,
                        simulator.wp_init_velocities
                    )
                    
                    for j in range(1, cfg.train_frame):
                        simulator.set_controller_target(j)
                        
                        if simulator.object_collision_flag:
                            simulator.update_collision_graph()
                        
                        if cfg.use_graph:
                            wp.capture_launch(simulator.graph)
                        else:
                            if cfg.data_type == "real":
                                with simulator.tape:
                                    simulator.step()
                                    simulator.calculate_loss()
                                simulator.tape.backward(simulator.loss)
                            else:
                                with simulator.tape:
                                    simulator.step()
                                    simulator.calculate_simple_loss()
                                simulator.tape.backward(simulator.loss)
                        
                        collide_optimizer.step()
                        
                        loss = wp.to_torch(simulator.loss, requires_grad=False)
                        iter_losses.append(loss.item())
                        
                        if cfg.use_graph:
                            simulator.tape.zero()
                        else:
                            simulator.tape.reset()
                        
                        simulator.clear_loss()
                        simulator.set_init_state(
                            simulator.wp_states[-1].wp_x,
                            simulator.wp_states[-1].wp_v
                        )
                    
                    avg_loss = np.mean(iter_losses)
                    print(f"Collide iter {opt_iter+1}/5 - Avg Loss: {avg_loss:.10f}")
        else:
            # Validation mode
            _, _, _, _, _, _, _, loss, chamfer_loss, track_loss = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=False,
                set_object_point=False
            )
            total_loss_val = loss

        print("Backward/Overhead time : {}".format(time() - st))
        mean_loss = np.sum(total_loss_val) if isinstance(total_loss_val, list) else total_loss_val

        if mode == 'train':
            return mean_loss, [current_mech_info] if 'current_mech_info' in locals() else None
        else:
            return mean_loss, None, None