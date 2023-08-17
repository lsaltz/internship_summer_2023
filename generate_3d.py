import bpy
import bmesh
import os
import numpy as np
import camera as c
from random import randint
from mathutils import Vector, Matrix, Euler
from math import radians
from scipy.spatial.transform import Rotation as R
from curve_fitting import BezierBasedDetection, Bezier
from bpy_extras.io_utils import unpack_list
class Generate_3D:   
    def __init__(self, full_arr, y, rad, ts, curve, num_pix):
        self.full_arr = full_arr
        self.rad = rad
        self.ts = ts
        self.curve = curve
        
    def create_mesh(self):
        height = self.get_translation()
        rad1, rad2 = self.get_radii()
        a = self.get_angle()
        
        bpy.data.objects.remove(bpy.data.objects['Cube'])

        crv = bpy.data.curves.new('crv', 'CURVE')
        crv.dimensions = '3D'
        crv.dimensions = '3D'
        crv.fill_mode = 'FULL'
        crv.bevel_depth = 0.045
            
        obj = bpy.data.objects.new('crv', crv)
        bpy.context.scene.collection.objects.link(obj)         
        spline = crv.splines.new(type='BEZIER')
        spline.bezier_points.add(len(self.full_arr)-1)
        for i in range(len(self.full_arr)):
           
            x, y, z = self.full_arr[i]
            spline.bezier_points[i].co = self.full_arr[i]
            #spline.bezier_points.handle_right = self.full_arr[i+1]
          
            spline.bezier_points[i].radius = self.rad[i]*10
            #print(spline.points[i].radius)    
        crv.resolution_u = 1  
   
    
       def set_y(self):
            camera = c.PinholeCameraModel()
            camera.from_npz(p.cam)
            assert isinstance(camera, c.PinholeCameraModel)
            for i in range(len(self.full_arr)):
                real_height = (c.getDeltaY(self.y, self.depth)
                self.full_arr[i][1] = real_height
                
       def set_rad(self):
            camera = c.PinholeCameraModel()
            camera.from_npz(p.cam)
            assert isinstance(camera, c.PinholeCameraModel)
            for i in range(len(self.full_arr)):
                real_rad = (c.getDeltaX(self.num_pix, self.depth)
                self.full_arr[i][0] = real_rad          
        """
        for i in range(len(height)):
            bm = bmesh.new()
            m = Matrix()
            
            bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, radius1=rad1[i], radius2=rad2[i], depth=height[i][1], segments=12, matrix=m.Translation(matr), calc_uvs=False)
            ver = bm.verts[:]
            bmesh.ops.rotate(bm, verts=ver, cent=self.full_arr[i], matrix=Matrix.Rotation(radians(90), 3, 'X'))
            
            ver = bm.verts[:]
            bmesh.ops.rotate(bm, verts=ver, cent=self.full_arr[i], matrix=Matrix.Rotation(np.cos(a[i]), 3, 'Z'))
            bmesh.ops.rotate(bm, verts=ver, cent=self.full_arr[i], matrix=Matrix.Rotation(np.sin(a[i]), 3, 'Y'))
            ver = bm.verts[:]
            
            mesh = bpy.data.meshes.new(f'cyl{i}')
            bm.to_mesh(mesh)
            bm.free()
            obj = bpy.data.objects.new(f'cyl{i}', mesh)
            
            
            

            #object.matrix_world = 
            #object.matrix_world = 
            #object.matrix_world = 
            #angle_in_degrees = 90
            #rot_mat = Matrix.Rotation(radians(angle_in_degrees), 4, 'X')   # you can also use as axis Y,Z or a custom vector like (x,y,z)
            #vert_loc = obj.location
            #mat_rot_x = Matrix.Rotation(x_angle[i], 4, vert_loc)
            #mat_rot_y = Matrix.Rotation(y_angle[i], 4, vert_loc)
            #mat_rot_z = Matrix.Rotation(radians(90), 4, vert_loc)
            #ob = bpy.data.objects[f'cyl{i}']
            #ob.rotation_axis_angle = (0, 0, height[i], 0)
            #bpy.ops.transform.rotate(value=radians(90), orient_axis='Y', orient_type='LOCAL')
            #ob.rotation_euler.rotate_axis("X", radians(90))
            #ob.rotation_euler.rotate_axis("Z", radians(90))
            #R = Euler((0, 0, radians(90))).to_matrix().to_4x4()
            #ob.matrix_local = R @ ob.matrix_local
            #print(ob.rotation_euler)
            
            #obj.matrix_local = mat_rot_y
            #obj.matrix_local = mat_rot_z
            
            #obj.rotation_euler = (radians(90), 0, 0)
            #obj.rotation_euler = (, 0)
            
            bpy.context.collection.objects.link(obj)    
        """   
            
            
        blend_file_path = bpy.data.filepath
        directory = os.path.dirname(blend_file_path)
        target_file = os.path.join(directory, 'thisisatest.obj')
        bpy.ops.export_scene.obj(filepath=target_file)
         
    def get_height(self):
        height = []
        for i in range(1, len(self.full_arr)-1):
            height.append(self.full_arr[i] - self.full_arr[i+1])
        return height
    
    def get_radii(self):
        rad1 = self.rad[0:-1]
        rad2 = self.rad[1:]
        return rad1, rad2
        
    def get_angle(self):
        x_angle = []
        y_angle = []
        a = []
        for i in range(1, len(self.full_arr)-1):
            a.append(np.arccos(np.clip(np.dot((self.curve.tangent(self.ts[i]) / np.linalg.norm(self.curve.tangent(self.ts[i]))), (self.curve.tangent(self.ts[i+1]) / np.linalg.norm(self.curve.tangent(self.ts[i+1])))), -1.0, 1.0)))
            x_angle.append(np.cos(a)-np.sin(a))
            y_angle.append(np.sin(a)+np.cos(a))
            
        return a

