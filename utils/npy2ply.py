import sys
import numpy as np
import open3d as o3d

inpath = str(sys.argv[1])
outpath = str(sys.argv[2])
xyzrgb = np.transpose(np.load(inpath))
print('Input point cloud consists of ' + str(xyzrgb[0].size) + ' XYZ points')
xyz = np.float32(np.transpose(np.row_stack((xyzrgb[0],xyzrgb[1],xyzrgb[2]))))
pcd = o3d.t.geometry.PointCloud(xyz)
print('Writing file ' + outpath)
o3d.t.io.write_point_cloud(outpath, pcd)
