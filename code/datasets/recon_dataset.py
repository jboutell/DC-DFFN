import torch.utils.data as data
from utils.general import *
import trimesh
from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import utils.general as utils
import numpy as np 
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
import torch 


class ReconDataSet(data.Dataset):

    def __init__(self,split,dataset_path,dist_file_name, num_of_points, num_of_tr_ts_points):

        self.num_of_tr_ts_points = num_of_tr_ts_points

        model = trimesh.load(dataset_path)
        if not type(model) == trimesh.PointCloud:
            model = utils.as_mesh(trimesh.load(dataset_path))
            if (model.faces.shape[0] > 0):
                #self.points = sample_surface(model,num_of_points)[0]
                sample = sample_surface(model, int(num_of_points*2))
                
                center = 0 * np.mean(sample[0], axis=0)
                scale = 1.0
                face_nrmls = model.face_normals[sample[1]]

                sample2 = sample_surface(model, num_of_points)
                pnts = sample2[0]
                triangles = []
                for tri in model.triangles:
                    a = Point_3((tri[0][0] - center[0]) / scale,
                            (tri[0][1] - center[1]) / scale,
                            (tri[0][2] - center[2]) / scale)
                    b = Point_3((tri[1][0] - center[0]) / scale,
                                (tri[1][1] - center[1]) / scale,
                                (tri[1][2] - center[2]) / scale)
                    c = Point_3((tri[2][0] - center[0]) / scale,
                                (tri[2][1] - center[1]) / scale,
                                (tri[2][2] - center[2]) / scale)
                    triangles.append(Triangle_3(a, b, c))
                tree = AABB_tree_Triangle_3_soup(triangles)
                sigmas = []
                ptree = cKDTree(pnts)
                i = 0
                for p in np.array_split(pnts, 100, axis=0):
                    d = ptree.query(p, 51)
                    sigmas.append(d[0][:, -1])

                    i = i + 1

                sigmas = np.concatenate(sigmas)
                sigmas_big = 0.3 * np.ones_like(sigmas)

                sample2 = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                        pnts + np.expand_dims(sigmas_big,-1) * np.random.normal(0.0, 1.0,size=pnts.shape)],
                                        axis=0)


                dists = []
                normals = []
                for np_query in tqdm(sample2):
                    cgal_query = Point_3(np_query[0].astype(np.double),
                                    np_query[1].astype(np.double),
                                    np_query[2].astype(np.double))

                    cp = tree.closest_point(cgal_query)
                    cp = np.array([cp.x(), cp.y(), cp.z()])
                    dist = np.sqrt(((cp - np_query) ** 2).sum(axis=0))
                    n = (np_query - cp) / dist
                    normals.append(np.expand_dims(n.squeeze(), axis=0))

                    dists.append(dist)
                dists = np.array(dists)
                normals = np.concatenate(normals, axis=0)
                self.dists = np.concatenate([sample2,normals, np.expand_dims(dists, axis=-1)], axis=-1)
                self.mnf_point = np.concatenate([sample[0], face_nrmls], axis=-1)
                self.npyfiles_mnfld = [dataset_path]
                self.scale = scale 
                self.center = center 
            else:
                if num_of_points < max((model.vertices).shape):
                    random_idx = [np.random.randint(0, num_of_points) for p in range(0, num_of_points)]
                    self.points = model.vertices[random_idx,:]
                else:
                    self.points = model.vertices
                    num_of_points = max((model.vertices).shape)

                self.points = self.points - self.points.mean(0,keepdims=True)
                scale = np.abs(self.points).max()
                center = center = 0 * np.mean(self.points, axis=0)
                self.points = self.points / scale

                sigmas = []

                ptree = cKDTree(self.points)

                for p in tqdm(np.array_split(self.points, 10, axis=0)):
                    d = ptree.query(p, np.int(51.0))
                    sigmas.append(d[0][:, -1])

                sigmas = np.concatenate(sigmas)
                sigmas = np.tile(sigmas, [num_of_points // sigmas.shape[0]])
                sigmas_big = 1.0 * np.ones_like(sigmas)
                pnts = np.tile(self.points, [num_of_points // self.points.shape[0], 1])

                sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                        pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                        axis=0)

                dists = []
                normals = []
                for np_query in tqdm(sample):
                    dist, idx = ptree.query(np_query)
                    n = (np_query - self.points[idx]) / dist
                    dists.append(dist)
                    normals.append(np.expand_dims(n.squeeze(), axis=0))

                dists = np.array(dists)
                normals = np.concatenate(normals, axis=0)
                self.dists = np.concatenate([sample,normals, np.expand_dims(dists, axis=-1)], axis=-1)
                self.mnf_point = np.concatenate([self.points, normals[:num_of_points,:]], axis=-1)
                self.npyfiles_mnfld = [dataset_path]
                self.scale = scale 
                self.center = center 
        else:
            if num_of_points < max((model.vertices).shape):
                random_idx = [np.random.randint(0, num_of_points) for p in range(0, num_of_points)]
                self.points = model.vertices[random_idx,:]
            else:
                self.points = model.vertices
                num_of_points = max((model.vertices).shape)

            self.points = self.points - self.points.mean(0,keepdims=True)
            scale = np.abs(self.points).max()
            center = center = 0 * np.mean(self.points, axis=0)
            self.points = self.points / scale

            sigmas = []

            ptree = cKDTree(self.points)

            for p in tqdm(np.array_split(self.points, 10, axis=0)):
                d = ptree.query(p, int(51.0))
                sigmas.append(d[0][:, -1])

            sigmas = np.concatenate(sigmas)
            sigmas = np.tile(sigmas, [num_of_points // sigmas.shape[0]])
            sigmas_big = 1.0 * np.ones_like(sigmas)
            pnts = np.tile(self.points, [num_of_points // self.points.shape[0], 1])

            sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                    pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                    axis=0)

            dists = []
            normals = []
            for np_query in tqdm(sample):
                dist, idx = ptree.query(np_query)
                n = (np_query - self.points[idx]) / dist
                dists.append(dist)
                normals.append(np.expand_dims(n.squeeze(), axis=0))

            dists = np.array(dists)
            normals = np.concatenate(normals, axis=0)
            self.dists = np.concatenate([sample,normals, np.expand_dims(dists, axis=-1)], axis=-1)
            self.mnf_point = np.concatenate([self.points, normals[:num_of_points,:]], axis=-1)
            self.npyfiles_mnfld = [dataset_path]
            self.scale = scale 
            self.center = center 
            print(scale, center)

    def __getitem__(self, index):
        point_set_mnlfld = torch.from_numpy(self.mnf_point).float()
        sample_non_mnfld = torch.from_numpy(self.dists).float()
        random_idx = (torch.rand(self.num_of_tr_ts_points**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld,0,random_idx)
        normal_set_mnfld = point_set_mnlfld[:,3:] 
        point_set_mnlfld = point_set_mnlfld[:,:3]# + self.normalization_params[index].float()

        random_idx = (torch.rand(self.num_of_tr_ts_points** 2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld, 0, random_idx)
        normalization_param1 = torch.from_numpy(np.asarray(self.center))
        normalization_param2 = torch.from_numpy(np.asarray(self.scale))
        normalization_param = {'center': normalization_param1, 'scale': normalization_param2}
                                         

        return point_set_mnlfld,normal_set_mnfld,sample_non_mnfld,normalization_param, index
    def __len__(self):
        return 1


