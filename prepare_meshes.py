import open3d as o3d
from datasets import get_datasets
from models.networks import HyperPointFlow
import torch
import numpy as np
import os
import pickle as pkl

from args import get_args
from models.atlas_networks import HyperNetwork, TargetNetwork
from scipy.spatial import Delaunay
from train_atlas import tile, generate_points_from_uniform_distribution


def get_grid(grain=5):
    grain = int(grain)
    grain = grain - 1  # to return grain*grain points
    # generate regular grid
    faces = []
    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain, 0])

    for i in range(1, int(grain + 1)):
        for j in range(0, (int(grain + 1) - 1)):
            faces.append([j + (grain + 1) * i,
                          j + (grain + 1) * i + 1,
                          j + (grain + 1) * (i - 1)])
    for i in range(0, (int((grain + 1)) - 1)):
        for j in range(1, int((grain + 1))):
            faces.append([j + (grain + 1) * i,
                          j + (grain + 1) * i - 1,
                          j + (grain + 1) * (i + 1)])

    return np.array(vertices)[:,:2]


def get_loader(args, split):
    if split=='train':
        tr_dataset, _ = get_datasets(args)
    else:
        _, tr_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        tr_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    ds_mean = torch.from_numpy(tr_dataset.all_points_mean).cuda()
    ds_std = torch.from_numpy(tr_dataset.all_points_std).cuda()
    return loader, ds_mean, ds_std


def main(args):
    hyper_dir = '/home/maciej/PycharmProjects/HyperFlow/meshes/hyper/'
    atlas_dir = '/home/maciej/PycharmProjects/HyperFlow/meshes/atlas/'
    model = HyperPointFlow(args)
    atlas = HyperNetwork(args)
    atlas = atlas.cuda()
    print("Resume Path:%s" % args.resume_atlas)
    checkpoint = torch.load(args.resume_atlas)
    atlas.load_state_dict(checkpoint['model'])
    model = model.cuda()
    split = 'test'
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    var = 0.01
    model.eval()
    model.var = var
    loader, ds_mean, ds_std = get_loader(args, split)
    triangles = 5
    file = open('/home/maciej/Pobrane/data_for_mesh/data_for_mesh_Triangultio_' + str(triangles) + '.pickle', 'rb')
    T = pkl.load(file)
    XX = np.load('/home/maciej/Pobrane/data_for_mesh/data_for_mesh_Triangultio_' + str(triangles) + '.npy')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XX)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(XX)
    mesh.triangles = o3d.utility.Vector3iVector(T)
    mesh.compute_vertex_normals()
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    mesh.orient_triangles()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=600, height=600)
    # vis.add_geometry(mesh)
    # vis.run()
    # vis.poll_events()
    # vis.update_renderer()
    # vis.destroy_window()
    point_sampled = torch.from_numpy(np.asarray(mesh.vertices)).float().unsqueeze(0).cuda()
    point_sampled = point_sampled.repeat(args.batch_size, 1, 1)
    for data in loader:
        idx_b, te_pc, cat = data['idx'], data[split + '_points'], data['cate_idx']
        print(" Current id: " + str(idx_b[-1]))
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        z = model.encode(te_pc)
        _, sample_pcs = model.sample(None, None, gpu=args.gpu, w=z, y=point_sampled)
        sample_pcs = sample_pcs * ds_std + ds_mean
        sample_pcs = sample_pcs.cpu().detach().numpy()
        for k in range(sample_pcs.shape[0]):
            pcl = o3d.geometry.PointCloud()
            pts = sample_pcs[0].reshape(-1, 3)
            pcl.points = o3d.utility.Vector3dVector(pts)
            mesh.vertices = pcl.points
            current_path = os.path.join(hyper_dir, data['sid'][k], "val")
            os.makedirs(current_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(current_path, data['mid'][k][4:] + ".obj"), mesh)
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=600, height=600)
            # vis.add_geometry(mesh)
            # vis.run()
            # vis.poll_events()
            # vis.update_renderer()
            # vis.destroy_window()
        N = 5000
        atlas_2d_points = 25
        grid = get_grid(5)
        target_networks_weights = atlas(z)
        _, out = model.sample(None, N, gpu=args.gpu, w=z)
        out_pc = torch.zeros((out.shape[0], out.shape[1], atlas_2d_points, out.shape[2])).cuda()
        # atlas_input = generate_points_from_uniform_distribution(size=(atlas_2d_points, 2), low=0, high=1,
        #                                                         norm=False).cuda()
        atlas_input = torch.from_numpy(grid).float().cuda()
        simplices = Delaunay(atlas_input.cpu().detach().numpy()).simplices

        # def swap_cols(arr, frm, to):
        #     arr[:, [frm, to]] = arr[:, [to, frm]]
        #
        # T_n = np.array(simplices)
        # swap_cols(T_n, 0, 2)
        # simplices = np.vstack((simplices, T_n))
        # sim_total = []
        # for k in range(N):
        #     sim_total.append(simplices+k*atlas_2d_points)
        # sim_total = np.concatenate(sim_total, axis=0)
        atlas_input = atlas_input.repeat(N, 1)
        for j, target_network_weight in enumerate(target_networks_weights):
            pc = out[j].detach()
            x_temp = torch.cat([tile(pc, dim=0, n_tile=atlas_2d_points), atlas_input], dim=1)
            target_network = TargetNetwork(args.zdim, target_network_weight).cuda()
            out_pc[j] = target_network(x_temp).reshape((pc.shape[0], atlas_2d_points, pc.shape[1]))
        out_pc = out_pc * ds_std + ds_mean
        out_pc = out_pc.cpu().detach().numpy()

        for k in range(sample_pcs.shape[0]):
            X_r = out[k].cpu().detach().numpy()
            XX_T = out_pc[k]
            T = simplices
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=500).fit(X_r)
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(X_r, np.ones(len(X_r)))
            indexes = neigh.kneighbors(X=km.cluster_centers_, n_neighbors=1, return_distance=False).T[0]

            def swap_cols(arr, frm, to):
                arr[:, [frm, to]] = arr[:, [to, frm]]

            T_n = np.array(T)
            swap_cols(T_n, 0, 2)
            selected = 0
            points = []
            sim_total = []
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=600, height=600)
            for i in indexes:
                XX = XX_T[i]
                if ((np.sum(np.max(XX, 0) - np.min(XX, 0))) < 0.43) and ((np.sum(np.max(XX, 0) - np.min(XX, 0))) > 0.05):
                    # points.append(XX)
                    # sim_total.append(np.vstack((T, T_n)) + selected * atlas_2d_points)
                    # selected = selected + 1
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(XX)
                    mesh.triangles = o3d.utility.Vector3iVector(np.vstack((T, T_n)))
                    mesh.compute_vertex_normals()
                    R = mesh.get_rotation_matrix_from_xyz((np.pi / 120, -np.pi / 5, np.pi / 2.2))
                    mesh.rotate(R, center=(0, 0, 0))
                    vis.add_geometry(mesh)
                    vis.update_geometry(mesh)
            vis.run()
            vis.poll_events()
            vis.update_renderer()
            vis.destroy_window()
            # sim_total = np.concatenate(sim_total, axis=0)
            # points = np.concatenate(points, axis=0)
            # mesha = o3d.geometry.TriangleMesh()
            # mesha.vertices = o3d.utility.Vector3dVector(points)
            # mesha.triangles = o3d.utility.Vector3iVector(sim_total)
            # mesha.compute_vertex_normals()
            # current_path = os.path.join(atlas_dir, data['sid'][k], "val")
            # os.makedirs(current_path, exist_ok=True)
            # o3d.io.write_triangle_mesh(os.path.join(current_path, data['mid'][k][4:] + ".obj"), mesha)
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=600, height=600)
            # vis.add_geometry(mesha)
            # vis.run()
            # vis.poll_events()
            # vis.update_renderer()
            # vis.destroy_window()


if __name__ == '__main__':
    args = get_args()
    main(args)