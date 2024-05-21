import os
from glob import glob
import trimesh
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from functools import partial
from tqdm import tqdm

"""
Reference: https://matplotlib.org/matplotblog/posts/custom-3d-engine/
"""


def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c,-s, 0, 0],
                      [ s, c, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ 1, 0, 0, 0],
                      [ 0, c,-s, 0],
                      [ 0, s, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(frame_v, MVP, F, norm=True, z_div=True):
    V = frame_v
    if norm:
        V = (V - (V.max(0) + V.min(0)) *0.5) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    V = V @ MVP.T
    if z_div:
        V /= V[:, 3].reshape(-1, 1)
    VF = V[F]
    return VF

def calc_norm_fv(fv):
    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-12)
    return norm
    
def plot_image_array(Vs, 
                     Fs, 
                     rot_list=None, 
                     size=6, 
                     norm=False, 
                     view_mode='p'
                     mode='mesh', 
                     linewidth=1, 
                     linestyle='solid', 
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                     logdir='.', 
                     name='000', 
                     save=False,
                    ):
    r"""
    Args:
        Vs (list): list of vertices [V, V, V, ...]
        Fs (list): list of face indices [F, F, F, ...]
        rot_list (list): list of euler angle [ [x,y,z], [x,y,z], ...]
        size (int): size of figure
        norm (bool): if True, normalize vertices
        view_mode (str): if 'p' use perspective, if 'o' use orthogonal camera
        mode (str): mode for rendering [mesh(wireframe), shade, normal]
        linewidth (float): line width for wireframe (kwargs for matplotlib)
        linestyle (str): line style for wireframe (kwargs for matplotlib)
        light_dir (np.array): light direction
        bg_black (bool): if True, use dark_background for plt.style
        logdir (str): directory for saved image
        name (str): name for saved image
        save (bool): if True, save the plot as image
    """
    if mode=='gouraud':
        print("currently WIP!: need to curl by z")
        
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)

        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -4) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        if view_mode=='p':
            proj  = perspective(55, 1, 1, 100)
        else:
            proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
            
        MVP   = proj @ model # view is identity
        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        if mode=='normal':
            C = calc_face_norm(V[F]) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            C = np.clip(C, 0, 1) if False else C * 0.5 + 0.5
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='shade':
            C = calc_face_norm(V[F]) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        elif mode=='gouraud':
#             I = np.argsort(Z)
#             V, F, vidx = get_new_mesh(V, F, I, invert=True)
            
            ### curling by normal
            C = calc_norm(V, F, mode='v') #@ model[:3,:3].T
            NI = np.argwhere(C[:,2] > 0.0).squeeze()
            V, F, vidx = get_new_mesh(V, F, NI, invert=True)
            
            C = calc_norm(V, F,mode='v') #@ model[:3,:3].T
            
            #VV = (V-V.min()) / (V.max()-V.min())# world coordinate
            V = transform_vertices(V, MVP, F, norm, no_parsing=True)
            triangle_ = tri.Triangulation(V[:,0], V[:,1], triangles=F)
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            #VV = (V-V.min()) / (V.max()-V.min()) #screen coordinate
            #cmap = colors_to_cmap(VV)
            cmap = colors_to_cmap(C)
            zs = np.linspace(0.0, 1.0, num=V.shape[0])
            plt.tripcolor(triangle_, zs, cmap=cmap, shading='gouraud')
            
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor='black')
            
        if mode!='gouraud':
            ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()
    
def render_video(basedir="tmp", # mesh obj directory
                 savedir="tmp", # save path 
                 savename="tmp", # video name
                 size=3, # figure size
                 fps=30,
                 xrot=0,
                 yrot=0,
                 zrot=0,
                 light_dir=np.array([0,0,1]),
                 mode='mesh', 
                 linewidth=1,
                ):
    # make dirs
    os.makedirs(savedir, exist_ok=True)
    
    ## visualize
    basename = basedir
    
    meshes = sorted(glob.glob(os.path.join(basename, '*.obj')))
    tmp = trimesh.load(meshes[0])
    mesh_face = tmp.faces
    
    mesh_vtxs = []
    for mesh in meshes:
        tmp = trimesh.load(mesh)
        mesh_vtxs.append(tmp.vertices)
    mesh_vtxs = np.array(mesh_vtxs)
    
    num_meshes = len(mesh_vtxs)
    
    ## visualize
    fig = plt.figure(figsize=(size, size))
    _r = figsize[0] / figsize[1]
    fig_xlim = [-_r, _r]
    fig_ylim = [-1, +1]
    ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)

    ## MVP
    model = translate(0, 0, -2.5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    proj  = perspective(25, 1, 1, 100)
    MVP   = proj @ model # view is identity

    def render_mesh(ax, V, MVP, F):        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode=='shade':
            C = calc_norm_fv(V[F]) @ model[:3,:3].T
            I = np.argsort(Z) # -----------------------> depth sorting
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze() # --> culling w/ normal
            T, C = T[NI, :], C[NI, :]
            
            C = np.clip((C @ light_dir), 0, 1) # ------> cliping range 0 - 1
            C = C[:,np.newaxis].repeat(3, axis=-1)
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        ax.add_collection(collection)
    
    def update(V):
        # Cleanup previous collections
        for coll in ax.collections:
            coll.remove()

        # Render meshes for all views
        render_mesh(ax, V, MVP, mesh_face)
        
        return ax.collections
    
    #plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=mesh_vtxs, blit=True)
    
    bar = tqdm(total=num_meshes, desc="rendering")
    anim.save(
        f'{savedir}/{savename}.mp4', 
        fps=fps,
        progress_callback=lambda i, n: bar.update(1)
    )
    
def compute_average_distance(points):
    """
    Compute the average distance of each point from the origin in a set of points.

    :param points: A set of points.
    :return: Average distance.
    """
    distances = np.linalg.norm(points, axis=1)
    return np.mean(distances)

def kabsch_algorithm_with_scale(P, Q):
    """
    Extended Kabsch algorithm to find the optimal rotation, translation, and scaling
    that aligns two sets of points P and Q minimizing the RMSD.

    :param P: A set of points.
    :param Q: A set of corresponding points.
    :return: Rotation matrix (R), translation vector (t), scale factor (s).
    """

    # Calculate the centroids of the point sets
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the points around the origin
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Calculate the average distances from the origin
    avg_dist_P = compute_average_distance(P_centered)
    avg_dist_Q = compute_average_distance(Q_centered)

    # Calculate the scale factor
    s = avg_dist_Q / avg_dist_P

    # Scale the points
    P_scaled = P_centered * s

    # Compute the covariance matrix
    H = P_scaled.T @ Q_centered

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = -R @ (centroid_P * s) + centroid_Q

    return R, t, s
