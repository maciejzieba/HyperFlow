import open3d as o3d
from datasets import get_datasets
from args import get_args
from models.networks import HyperPointFlow
import os
import torch
import numpy as np
import torch.nn as nn
import sys

import pickle as pkl
import matplotlib as plt

from datasets import ShapeNet15kPointClouds


def custom_draw_geometry(pcd, path):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.rotate(180.0, 180.0)
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
    # vis.run()

    #vis.capture_screen_image(path)

    depth = vis.capture_depth_float_buffer(True)
    # image = vis.capture_screen_float_buffer(False)
    # vis.capture_depth_point_cloud(path)
    # sys.exit()
    vis.close()
    plt.pyplot.imsave(path, np.asarray(depth))
    vis.destroy_window()


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)


def main(args):
    mode = args.demo_mode
    folder = 'supplementary'
    N_samples = 20
    seed = 1234

    # Integrate mesh
    if mode == 0:
        use_meshes = True
        encode = True
        diff_t_values = True
        diff_r_values = True
        interpolate = False
        experiment = 'integrating'
        model_type = 'lognormal_mesh_r_001'
        args.use_sphere_dist = True
        r_set = [0.960, 0.973, 0.983, 0.992, 1.0, 1.008, 1.016, 1.026, 1.041]

    # Interpolate mesh
    if mode == 1:
        use_meshes = True
        encode = True
        diff_t_values = False
        diff_r_values = True
        interpolate = True
        experiment = 'interpolating'
        model_type = 'mesh_r'
        interpol_range = np.arange(0.0, 1.01, 0.2)
        r_set = [0.960, 0.973, 0.983, 0.992, 1.0, 1.008, 1.016, 1.026, 1.041]

    # Sample mesh
    if mode == 2:
        use_meshes = True
        encode = False
        diff_t_values = False
        diff_r_values = True
        interpolate = False
        experiment = 'sample_r'
        model_type = 'mesh'
        args.use_sphere_dist = True
        r_set = [0.960, 0.973, 0.983, 0.992, 1.0, 1.008, 1.016, 1.026, 1.041]

    # Integrate logNormal
    if mode == 3:
        use_meshes = False
        encode = True
        diff_t_values = True
        diff_r_values = False
        interpolate = False
        experiment = 'integrating'
        model_type = 'point_lognormal'

    # Interpolate lognormal
    if mode == 4:
        use_meshes = False
        encode = True
        diff_t_values = False
        diff_r_values = False
        interpolate = True
        experiment = 'interpolating'
        model_type = 'point_lognormal'
        interpol_range = np.arange(0.0, 1.01, 0.2)

    # Sample logNormal
    if mode == 5:
        use_meshes = False
        encode = False
        diff_t_values = False
        diff_r_values = False
        interpolate = False
        experiment = 'sample'
        model_type = 'log_normal'

    _, te_dataset = get_datasets(args)
    N_total = te_dataset.all_points.shape[0]

    # idx = 60
    if encode:
        np.random.seed(seed=seed)
        ids = list(np.random.choice(N_total-1, N_samples, replace=False))
    else:
        ids = [k for k in range(N_samples)]

    if use_meshes:
        # T_set = [3, 4, 5, 6, 7]
        T_set = [4]
        var = 0.0
    else:
        T_set = [0]
        var = 0.001

    model = HyperPointFlow(args)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)

    model = model.cuda()
    model.eval()
    model.var = var

    for triangles in T_set:
        if use_meshes:
            path = os.path.join(folder, experiment, model_type, 'Triangultio_' + str(triangles), args.cates[0])
        else:
            path = os.path.join(folder, experiment, model_type, args.cates[0])
        for idx in ids:
            path_data = os.path.join(path, str(idx), "data")
            path_plots = os.path.join(path, str(idx), "plots")
            os.makedirs(path_data, exist_ok=True)
            os.makedirs(path_plots, exist_ok=True)
            if encode:
                test_numpy = te_dataset.test_points[idx]
                pcd = o3d.geometry.PointCloud()
                test_numpy = te_dataset.all_points_std * test_numpy + te_dataset.all_points_mean
                pcd.points = o3d.utility.Vector3dVector(test_numpy.reshape(-1, 3))
                custom_draw_geometry(pcd, os.path.join(path_plots, "example_ref.png"))
                np.save(os.path.join(path_data, "example_ref.npy"), test_numpy.reshape(-1, 3))
                if interpolate:
                    idx2 = ids[(ids.index(idx) + 1) % len(ids)]
                    test_numpy = te_dataset.test_points[idx2]
                    pcd = o3d.geometry.PointCloud()
                    test_numpy = te_dataset.all_points_std * test_numpy + te_dataset.all_points_mean
                    pcd.points = o3d.utility.Vector3dVector(test_numpy.reshape(-1, 3))
                    custom_draw_geometry(pcd, os.path.join(path_plots, "example_ref" + str(idx2) + ".png"))
                    np.save(os.path.join(path_data, "example_ref" + str(idx2) + ".npy"), test_numpy.reshape(-1, 3))
            else:
                z_sampled = model.sample_gaussian((args.batch_size, model.zdim)).cuda()

            if use_meshes:
                file = open('pretrained_model/demo/data_for_mesh/data_for_mesh_Triangultio_' + str(triangles) + '.pickle', 'rb')
                T = pkl.load(file)
                XX = np.load('pretrained_model/demo/data_for_mesh/data_for_mesh_Triangultio_' + str(triangles) + '.npy')

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(XX)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(XX)
                mesh.triangles = o3d.utility.Vector3iVector(T)
                mesh.compute_vertex_normals()
                mesh.compute_vertex_normals()
                mesh.normalize_normals()
                mesh.orient_triangles()
                if diff_r_values:
                    point_sampled_set = []
                    for r in r_set:
                        point_sampled_temp = r*np.asarray(mesh.vertices)
                        point_sampled_set.append(torch.from_numpy(point_sampled_temp).float().unsqueeze(0).cuda())
                else:
                    point_sampled = torch.from_numpy(np.asarray(mesh.vertices)).float().unsqueeze(0).cuda()
            else:
                if args.use_sphere_dist:
                    point_sampled = model.sample_lognormal((args.batch_size, args.num_sample_points, 3)).cuda()
                else:
                    point_sampled = model.sample_gaussian((args.batch_size, args.num_sample_points, 3)).cuda()

            if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
                mean = np.load(args.resume_dataset_mean)
                std = np.load(args.resume_dataset_std)
                te_dataset.renormalize(mean, std)
            ds_mean = torch.from_numpy(te_dataset.all_points_mean).cuda()
            ds_std = torch.from_numpy(te_dataset.all_points_std).cuda()
            if diff_t_values:
                t_values = [np.sqrt(t)*model.point_cnf.chain[1].sqrt_end_time.data for t in np.arange(0.0, 1.00001, 0.025)]
                t_values[0] = 0.01*model.point_cnf.chain[1].sqrt_end_time.data
            else:
                t_values = [model.point_cnf.chain[1].sqrt_end_time.data]
            all_sample = []
            with torch.no_grad():
                for t in t_values:
                    model.point_cnf.chain[1].sqrt_end_time.data = t
                    B = args.batch_size
                    N = args.num_sample_points
                    if encode:
                        if interpolate:
                            sample_1 = te_dataset.test_points[idx]
                            sample_2 = te_dataset.test_points[idx2]
                            z_1 = model.encode(torch.from_numpy(sample_1).float().unsqueeze(0).cuda())
                            z_2 = model.encode(torch.from_numpy(sample_2).float().unsqueeze(0).cuda())
                            out_pc = []
                            for k in range(interpol_range.shape[0]):
                                if diff_r_values:
                                    for point_sampled in point_sampled_set:
                                        out_pc.append(
                                            model.decode((1 - interpol_range[k]) * z_1 + interpol_range[k] * z_2,
                                                         N, y=point_sampled)[1])
                                else:
                                    out_pc.append(model.decode((1 - interpol_range[k]) * z_1 + interpol_range[k] * z_2,
                                                               N, y=point_sampled)[1])
                            out_pc = torch.cat(out_pc, dim=0)
                        elif diff_r_values:
                            out_pc = []
                            for point_sampled in point_sampled_set:
                                sample = te_dataset.test_points[idx]
                                out_pc.append(model.reconstruct(torch.from_numpy(sample).float().unsqueeze(0).cuda(),
                                                           N, y=point_sampled))
                            out_pc = torch.cat(out_pc, dim=0)
                        else:
                            sample = te_dataset.test_points[idx]
                            out_pc = model.reconstruct(torch.from_numpy(sample).float().unsqueeze(0).cuda(),
                                                       N, y=point_sampled)

                    else:
                        if diff_r_values:
                            out_pc = []
                            for point_sampled in point_sampled_set:
                                _, out_pc_temp = model.sample(B, N, gpu=args.gpu, w=z_sampled, y=point_sampled)
                                out_pc.append(out_pc_temp)
                            out_pc = torch.cat(out_pc, dim=0)
                        else:
                            _, out_pc = model.sample(B, N, w=z_sampled, y=point_sampled)
                    out_pc = out_pc * ds_std + ds_mean
                    all_sample.append(out_pc)
            sample_pcs = torch.cat(all_sample, dim=0).cpu().detach().numpy()
            print("Generation sample size:(%s, %s, %s)" % sample_pcs.shape)

            # Visualize the demo
            pcl = o3d.geometry.PointCloud()
            for i in range(int(sample_pcs.shape[0])):
                print("Visualizing: %03d/%03d" % (i, sample_pcs.shape[0]))
                pts = sample_pcs[i].reshape(-1, 3)
                pcl.points = o3d.utility.Vector3dVector(pts)
                if diff_r_values:
                    r_id = i % len(r_set)
                    t_id = int(i/len(r_set))
                    if use_meshes:
                        mesh.vertices = pcl.points
                        custom_draw_geometry(mesh, os.path.join(path_plots, "example_mesh_r_id_" + str(r_id) +
                                                                "t_id_" + str(t_id) + ".png"))

                    custom_draw_geometry(pcl, os.path.join(path_plots, "example_points_r_id_" + str(r_id) +
                                                                "t_id_" + str(t_id) + ".png"))
                    np.save(os.path.join(path_data, "example_points_r_id_" + str(r_id) + "t_id_" +
                                         str(t_id) + '.npy'), pts)
                else:
                    if use_meshes:
                        mesh.vertices = pcl.points
                        custom_draw_geometry(mesh, os.path.join(path_plots, "example_mesh" + str(i) + ".png"))

                    custom_draw_geometry(pcl, os.path.join(path_plots, "example_points" + str(i) + ".png"))
                    np.save(os.path.join(path_data, 'example_points' + str(i) + '.npy'), pts)


if __name__ == '__main__':
    args = get_args()
    main(args)
