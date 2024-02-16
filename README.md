# Matplotlib render
render 3D mesh with matplotlib library

Reference: https://matplotlib.org/matplotblog/posts/custom-3d-engine/

# Requirements
```
pip install -r requirements.txt
```


# How to use
```python
from matplotlib_render import *
import trimesh
mesh = trimesh.load('your_mesh_file.obj')

# figuer size
SIZE = 2

# a mesh that you wanna render
v_list=[mesh.vertices]
f_list=[mesh.faces]

# xyz Euler angle to rotate the mesh
rot_list=[[0,0,0]]*len(v_list)

plot_image_array(v_list, f_list, rot_list=rot_list, size=SIZE)
plot_image_array(v_list, f_list, rot_list=rot_list, size=SIZE, mode='normal')
plot_image_array(v_list, f_list, rot_list=rot_list, size=SIZE, mode='shade')
```
<img src="https://github.com/chacorp/matplotlib_render/blob/main/demo.png" width="256" height="740" />
