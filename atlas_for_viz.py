import open3d as o3d
from datasets import get_datasets, synsetid_to_cate
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import HyperPointFlow
from models.atlas_networks import HyperNetwork, TargetNetwork
import os
import torch
import numpy as np
import torch.nn as nn


from train_atlas import tile, generate_points_from_uniform_distribution
from scipy.spatial import Delaunay


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


def get_data_viz(model, atlas, args):
    save_dir = os.path.dirname(args.resume_atlas)
    B = 10
    N = 10000
    atlas_2d_points = 25
    grid = get_grid(5)
    z, out = model.sample(B, N, gpu=args.gpu)
    target_networks_weights = atlas(z)
    out_pc = torch.zeros((out.shape[0], out.shape[1], atlas_2d_points, out.shape[2])).cuda()
    #atlas_input = generate_points_from_uniform_distribution(size=(atlas_2d_points, 2), low=0, high=1, norm=False).cuda()
    atlas_input = torch.from_numpy(grid).float().cuda()
    simplices = Delaunay(atlas_input.cpu().detach().numpy()).simplices
    atlas_input = atlas_input.repeat(N, 1)
    for j, target_network_weight in enumerate(target_networks_weights):
        pc = out[j].detach()
        x_temp = torch.cat([tile(pc, dim=0, n_tile=atlas_2d_points), atlas_input], dim=1)
        target_network = TargetNetwork(args.zdim, target_network_weight).cuda()
        out_pc[j] = target_network(x_temp).reshape((pc.shape[0], atlas_2d_points, pc.shape[1]))
        # denormalize
    _, te_dataset = get_datasets(args)
    m = torch.from_numpy(te_dataset.all_points_mean).cuda()
    s = torch.from_numpy(te_dataset.all_points_std).cuda()
    m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
    s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
    out_pc = out_pc * s + m
    out = out * s + m
    out_pc = out_pc.cpu().detach().numpy()
    out = out.cpu().detach().numpy()
    np.save(os.path.join(save_dir, "atlas_points.npy"), out_pc)
    np.save(os.path.join(save_dir, "hyperflow_points.npy"), out)
    np.save(os.path.join(save_dir, "triangles.npy"), simplices)

    return out_pc, out, simplices


def main(args):
    model = HyperPointFlow(args)
    model = model.cuda()
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    atlas = HyperNetwork(args)
    atlas = atlas.cuda()
    print("Resume Path:%s" % args.resume_atlas)
    checkpoint = torch.load(args.resume_atlas)
    atlas.load_state_dict(checkpoint['model'])
    # vars = [0.01, 0.001, 0.0001, 0.0]
    var = 0.01
    model.eval()
    model.var = var
    with torch.no_grad():
        out_pc, out, simplices = get_data_viz(model, atlas, args)
    index = 3
    X_r = out[index]
    XX_T = out_pc[index]
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

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=600)
    for i in indexes:
        XX = XX_T[i]
        if ((np.sum(np.max(XX, 0) - np.min(XX, 0))) < 0.43) and ((np.sum(np.max(XX, 0) - np.min(XX, 0))) > 0.05):
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
    save_dir = os.path.dirname(args.resume_atlas)
    vis.capture_screen_image(os.path.join(save_dir, "3d_vis_" + str(index) + ".png"))
    vis.destroy_window()


if __name__ == '__main__':
    args = get_args()
    main(args)
