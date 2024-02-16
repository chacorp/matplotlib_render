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
    
def plot_image_array(Vs, Fs, rot_list=None, size=6, norm=False, mode='mesh', linewidth=1, linestyles='solid'): 
    num_meshes = len(Vs)
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=True)

        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -6) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        #proj  = perspective(30, 1, 1, 100)
        proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
        MVP   = proj @ model # view is identity
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        if mode=='normal':
            C = calc_norm_fv(V[F]) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            C = np.clip(C, 0, 1) if False else C * 0.5+ 0.5
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C, linestyles=linestyles)
        elif mode=='shade':
            C = calc_norm_fv(V[F]) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            C = (C @ np.array([0,0,1]))[:,np.newaxis].repeat(3,1)
            print(C.shape)
            C = np.clip(C, 0, 1)
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C, linestyles=linestyles)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor='black')
        
        ax.add_collection(collection)
    plt.show()
    
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
