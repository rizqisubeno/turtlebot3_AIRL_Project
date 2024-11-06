#!/usr/bin/python3.11

import bpy
import shutil
import glob, os

# Function to scale the imported .dae object
def scale_dae_object(filepath, scale_factors, output_filepath):
    # Clear all objects in the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the .dae file
    bpy.ops.wm.collada_import(filepath=filepath,
                              auto_connect=True,
                              find_chains= True,
                              fix_orientation=True,
                              import_units=True)

    bpy.ops.transform.resize(value=scale_factors)
    bpy.ops.object.transform_apply(scale=True)
    # # Get all imported objects
    # imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    # # Apply scaling to each imported object
    # for obj in imported_objects:
    #     obj.scale = scale_factors

    # Export the scaled object to a new .dae file
    bpy.ops.wm.collada_export(filepath=output_filepath)

# Folder paths
input_folderpath = "./original"  # Change this to the path of your input .dae file
output_folderpath = "scaled"  # Change this to the path of your output .dae file

# # Scale the .dae object
# scale_dae_object(input_filepath, scale_factors, output_filepath)

def get_all_dae_files(root_folder):
    dae_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.dae'):
                dae_files.append(os.path.join(root, file))
    return dae_files

# Get all .dae file paths
dae_file_paths = get_all_dae_files(input_folderpath)

# Print the paths
for path in dae_file_paths:
    if "visual" not in path:
        path_list = path.split('/')
        path_list.pop(1)
        path_list.insert(1,output_folderpath)

        # copy material texture
        path_input_check_material = '/'.join(path.split('/')[:-2]) + "/materials"
        path_output_check_material = '/'.join(path_list[:-2]) + "/materials"

        # destination scaled
        path_output_check = '/'.join(path_list[:-1])
        path_output = '/'.join(path_list)[:-4]

        # print(path_output)
        if (not os.path.exists(path_output_check)):
            os.makedirs(path_output_check)

        if (not os.path.exists(path_output_check_material)):
            shutil.copytree(path_input_check_material,path_output_check_material)

        if "ground" not in path.lower():
            # Scaling factors (x, y, z)
            scale_factors = (0.375, 0.375, 0.375)
        else:
            scale_factors = (0.375, 0.375, 1.0)
            print("warning ground detected")
        scale_dae_object(path, scale_factors, path_output)