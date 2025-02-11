import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from matplotlib.colors import ListedColormap
import os

def create_geometry_3d(cube_size, air_size, wall_thickness):
    # Initialize the cube with walls
    geometry = np.zeros((cube_size,100,100), dtype=int)

    # Set the air region
    air_start = (cube_size - air_size) // 2
    air_end = air_start + air_size
# Set the air region inside the cube
    geometry[
        air_start : 200,
        air_start : 100,
        air_start + wall_thickness : 100
    ] = 1  # Air is represented by 1

    return geometry, air_start, air_end

import numpy as np
import random

def add_random_shape_3d(i, geometry, air_start, air_end, wall_thickness, cube_size=200):
    permittivity_object = random.uniform(4, 40.0)
    size_w = random.randint(20, 40)
    size_h = random.randint(20, 40)
    size_d = random.randint(20, 40)  # Depth for rectangular shapes
    objwall_gap = 10  # Gap between object and wall

    # Get geometry shape limits
    x_max, y_max, z_max = geometry.shape

    # Calculate the center of the geometry
    center_z = z_max // 2  # Middle along z-axis

    # Adjust start positions to be closer to the center while avoiding out-of-bounds errors
    y_start = random.randint(
        wall_thickness + objwall_gap + size_w // 2,
        wall_thickness + objwall_gap - size_w //2 + y_max//2 - 5,
    )
    
    x_start = random.randint(air_start + objwall_gap + size_w//2 + x_max//4, 3*x_max//4 - objwall_gap - size_w//2)

    print(f"x_start: {x_start}, y_start: {y_start}, center_z: {center_z}")

    shape = random.choice(["cube", "sphere", "triangle_cube", "rectangle_cube"])

    if shape == "cube":
        # Ensure indices stay in bounds
        x_end = min(x_start + size_w, x_max)
        y_end = min(y_start + size_w, y_max)
        z_start = max(center_z - size_w // 2, 0)
        z_end = min(center_z + size_w // 2, z_max)

        geometry[x_start:x_end, y_start:y_end, z_start:z_end] = int(i) + 2

    elif shape == "sphere":
        radius = size_w // 2
        x_center, y_center, z_center = x_start + radius, y_start + radius, center_z

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                for z in range(-radius, radius + 1):
                    if x**2 + y**2 + z**2 <= radius**2:
                        xi, yi, zi = x_center + x, y_center + y, z_center + z
                        if 0 <= xi < x_max and 0 <= yi < y_max and 0 <= zi < z_max:
                            geometry[xi, yi, zi] = int(i) + 2

    elif shape == "triangle_cube":
        for x in range(size_w):
            for y in range(size_w - x):  # Sloped triangular shape
                for z in range(size_w - x):
                    xi, yi, zi = x_start + x, y_start + y, center_z + z
                    if 0 <= xi < x_max and 0 <= yi < y_max and 0 <= zi < z_max:
                        geometry[xi, yi, zi] = int(i) + 2

    elif shape == "rectangle_cube":
        # Ensure indices stay in bounds
        x_end = min(x_start + size_w, x_max)
        y_start_adj = max(y_start - size_h // 2, 0)
        y_end = min(y_start + size_h // 2, y_max)
        z_end = min(center_z + size_d, z_max)

        geometry[x_start:x_end, y_start_adj:y_end, center_z:z_end] = int(i) + 2

    return permittivity_object, shape, geometry

# def visualize_top_view(geometry, wall_color, air_color, f_color, s_color, t_color):
def visualize_top_view(geometry, **kwargs):
    top_view = geometry.max(axis=1)  # Maximum projection along z-axis
    cmap = ListedColormap([
        color for color in [
            kwargs.get('wall_color'), 
            kwargs.get('air_color'), 
            kwargs.get('f_color'), 
            kwargs.get('s_color'), 
            kwargs.get('t_color')
        ] if color is not None
    ])
    plt.figure(figsize=(10, 10))
    plt.imshow(top_view, cmap=cmap,origin='lower')
    plt.title('Top View of 3D Geometry')
    plt.axis('off')
    plt.show()

def save_top_view(filename, geometry, cube_size, **kwargs):
    # top_view = geometry.max(axis=2)  # Maximum projection along z-axis
    cmap = ListedColormap([
        color for color in [
            kwargs.get('wall_color'), 
            kwargs.get('air_color'), 
            kwargs.get('f_color'), 
            kwargs.get('s_color'), 
            kwargs.get('t_color')
        ] if color is not None
    ])
    # Create a 3D square array (cube) with ones
    square_size = max(geometry.shape)
    square = np.ones((square_size, square_size, geometry.shape[2]), dtype=int)

    # Insert geometry into the square (assuming it fits within the square)
    square[:geometry.shape[0], :geometry.shape[1], :geometry.shape[2]] = geometry

    # Generate the top view by taking the maximum projection along the z-axis
    top_view = square.max(axis=2)

    top_view = np.rot90(top_view, k=1, axes=(1, 0))  # Rotate 90 degrees to match the visualization

    plt.figure(figsize=(10, 10))
    plt.imshow(top_view, cmap=cmap, origin='lower')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, format='png', dpi=cube_size / 10, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_parameters(filename, **params):
    if os.path.exists(filename):
        existing_data = np.load(filename, allow_pickle=True)
        all_params = list(existing_data['params'])
    else:
        all_params = []

    all_params.append(params)

    with open(filename, 'wb') as f:
        np.savez(f, params=all_params)
import h5py
import numpy as np

def rotate_3d_array_z_plus_90(array):
    """
    Rotates a 3D array 90 degrees counterclockwise around the Z+ axis.
    
    Args:
        array (numpy.ndarray): The input 3D array of shape (X, Y, Z).
    
    Returns:
        numpy.ndarray: The rotated 3D array.
    """
    return np.flip(array.transpose(1, 0, 2), axis=0)

def save_geometry_to_h5(filename, geometry, **metadata):
    """
    Save the geometry data and metadata to an HDF5 file.
    
    Parameters:
        filename (str): Path to the HDF5 file.
        geometry (np.ndarray): 3D geometry data to save.
        metadata (dict): Additional metadata to store in the HDF5 file.
    """
    with h5py.File(filename, 'w') as h5file:
        # print(geometry.shape)
        geometry = np.swapaxes(geometry, 1, 2)  # Swap y and z axis
        # print(geometry.shape)
        # Save geometry data
        h5file.create_dataset('data', data=geometry, compression="gzip")

        # Save metadata
        for key, value in metadata.items():
            if isinstance(value, list):
                h5file.attrs[key] = str(value)  # Store lists as strings
            else:
                h5file.attrs[key] = value

if __name__ == '__main__':    
    # Predefined colors
    wall_color = [1, 1, 0]   # Wall color
    air_color = [1, 1, 1]    # Air color
    f_color = [1, 0, 0]      # First shape color
    s_color = [0, 1, 0]      # Second shape color
    t_color = [0, 0, 1]      # Third shape color
    # Argument parsing
    parser = argparse.ArgumentParser(description='Generate and visualize 3D geometries with random shapes.')
    parser.add_argument('--start', type=int, default=0, help='Starting index for geometry generation')
    parser.add_argument('--end', type=int, default=10, help='Ending index for geometry generation')
    args = parser.parse_args()

    args.n = args.end + 1 - args.start

    if not os.path.exists('./Geometry_3D/Object'):
        os.makedirs('./Geometry_3D/Object')
    if not os.path.exists('./Geometry_3D/Base'):
        os.makedirs('./Geometry_3D/Base')

    for i in range(args.n):
        cube_size = 200
        wall_thickness = random.randint(15, 30)

        # Define wall materials with permittivity and conductivity
        wall_materials = {
            "Concrete": {"permittivity": 5.24, "conductivity": 0.001},
            "Brick": {"permittivity": 3.91, "conductivity": 0.002},
            "Plasterboard": {"permittivity": 2.73, "conductivity": 0.0005},
            "Wood": {"permittivity": 1.99, "conductivity": 0.0002},
            "Glass": {"permittivity": 6.31, "conductivity": 0.00001},
        }

        # Variance factor for permittivity
        variance_factor = 0.1

        # Randomly select a wall material
        wall_material = random.choice(list(wall_materials.keys()))

        # Get the base permittivity and conductivity
        base_permittivity = wall_materials[wall_material]["permittivity"]
        conductivity = wall_materials[wall_material]["conductivity"]

        # Add variability to permittivity
        variance = base_permittivity * variance_factor
        permittivity_wall = round(random.uniform(base_permittivity - variance, base_permittivity + variance), 2)

        filename = f'./Geometry_3D/Object/geometry{i + args.start}.png'
        basename = f'./Geometry_3D/Base/base{i + args.start}.png'
        params_filename = f'./Geometry_3D/params_{args.start}_{args.end}.npz'
        h5_filename = f'./Geometry_3D/Object/geometry_{i + args.start}.h5'
        h5_basename = f'./Geometry_3D/Base/base_{i + args.start}.h5'
        geometry, air_start, air_end = create_geometry_3d(cube_size, cube_size, wall_thickness)
        save_top_view(basename, geometry, cube_size, wall_color = wall_color, air_color = air_color)
        
        # Save geometry to HDF5
        save_geometry_to_h5(
            h5_basename,
            geometry,
            cube_size=cube_size,
            wall_thickness=wall_thickness,
            wall_color=wall_color,
            air_color=air_color,
            permittivity_wall=permittivity_wall
        )
        per_obj_arr = []
        shape_arr = []
        num_objects = random.randint(1, 1)
        for j in range(num_objects):
            per_obj, shape, geometry = add_random_shape_3d(j, geometry, air_start, air_end, wall_thickness)
            per_obj_arr.append(per_obj)
            shape_arr.append(shape)


        save_top_view(
            filename, 
            geometry, 
            cube_size, 
            wall_color=wall_color, 
            air_color=air_color, 
            f_color=f_color if num_objects >= 1 else None, 
            s_color=s_color if num_objects >= 2 else None, 
            t_color=t_color if num_objects >= 3 else None
        )

        save_parameters(
            params_filename,
            shape=shape_arr,
            cube_size=cube_size,
            wall_thickness=wall_thickness,
            wall_color=wall_color,
            air_color=air_color,
            object_color=[f_color, s_color, t_color],
            permittivity_object=per_obj_arr,
            permittivity_wall=permittivity_wall,
            conductivity_wall=conductivity,
        )

        # Save geometry to HDF5
        save_geometry_to_h5(
            h5_filename,
            geometry,
            shape=shape_arr,
            cube_size=cube_size,
            wall_thickness=wall_thickness,
            wall_color=wall_color,
            air_color=air_color,
            object_color=[f_color, s_color, t_color],
            permittivity_object=per_obj_arr,            
            permittivity_wall=permittivity_wall,
            conductivity_wall=conductivity,
        )