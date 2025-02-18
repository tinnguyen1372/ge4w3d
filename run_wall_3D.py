from gprMax.gprMax import api
from gprMax.receivers import Rx
from tools.outputfiles_merge import merge_files
from tools.plot_Bscan import get_output_data, mpl_plot as mpl_plot_Bscan 
from tools.plot_Ascan import mpl_plot as mpl_plot_Ascan
from gprMax.receivers import Rx
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import random
import os

from scipy.ndimage import zoom

class Wall_Func():
    def __init__(self, args) -> None:
        self.args = args
        self.i = args.i
        self.restart = 1
        # self.num_scan = 20
        self.num_scan = 50

        self.resol = 0.005
        self.time_window = 40e-9
        self.square_size = args.square_size
        self.wall_thickness = args.wall_thickness
        # self.wall_height = args.wall_height
        self.wall_permittivity = args.wall_permittivity
        self.object_permittivity = args.object_permittivity
            
        # self.object_width = args.obj_width
        # self.object_height = args.obj_height
        self.src_to_wall = 0.10
        self.src_to_rx = 0.05
        # Geometry load
        self.base = os.getcwd() + '/Geometry_3D/Base'
        self.basefile = self.base + '/base_{}.h5'.format(i)
        self.geofolder = os.getcwd() + '/Geometry_3D/Object'
        self.geofile = self.geofolder + '/geometry_{}.h5'.format(i)

        # Data load
        self.pix =int(self.square_size/0.005)
        if not os.path.exists('./Input_3D'):
            os.makedirs('./Input_3D')        
        if not os.path.exists('./Input_3D/Base'):
            os.makedirs('./Input_3D/Base')
        if not os.path.exists('./Input_3D/Object'):
            os.makedirs('./Input_3D/Object')
        if not os.path.exists('./Output_3D'):
            os.makedirs('./Output_3D')
        if not os.path.exists('./Output_3D/Base'):
            os.makedirs('./Output_3D/Base')
        if not os.path.exists('./Output_3D/Object'):
            os.makedirs('./Output_3D/Object')
        if not os.path.exists('./Output_3D/WallObj'):
            os.makedirs('./Output_3D/WallObj')
        if not os.path.exists('./BaseImg_3D'):
            os.makedirs('./BaseImg_3D')
        if not os.path.exists('./ObjImg_3D'):
            os.makedirs('./ObjImg_3D')
        if not os.path.exists('./WallObj_3D'):
            os.makedirs('./WallObj_3D')

  
    def view_geometry(self):
        # self.preprocess(self.basefile)
        with h5py.File('./Geometry_3D/geometry_2d.h5', 'r') as f:
            data = f['data'][:]
        
        # Adjust large_array to match data's shape
        data = np.squeEye(data, axis=2)  # Remove any singleton dimensions, if needed
        large_array = np.full(data.shape, -1, dtype=int)
        # Override the values in large_array with data
        large_array[:data.shape[0], :data.shape[1]] = data

        # Mask the regions where the value is 1
        masked_data = ma.masked_where(large_array == -1, large_array)

        # Marker positions based on provided coordinates and scaling factor
        # marker_x, marker_y = 0.15 * data.shape[0] /3.33, 0.15 * data.shape[0] /3.33
        color_list = [
            (1.0, 1.0, 1.0),  # White for -1
            (1.0, 1.0, 0.0),  # Yellow for 0
            (1.0, 0, 0.0)   # Red for 1
        ]
        custom_cmap = ListedColormap(color_list, name="custom_cmap")
        # Plot the markers and masked data
        # plt.plot(marker_x, marker_y, marker='o', color='red', markersize=5)
        plt.imshow(masked_data, cmap='viridis')
        plt.axis('equal')
        plt.title("Geometry Visualization")
        plt.xlabel("X-axis (pixels)")
        plt.ylabel("Y-axis (pixels)")
        plt.show()

    def preprocess(self, filename):
        with h5py.File(filename, 'r') as f:
            data = f['data'][:]
        # Apply the transformation
        data = np.where(data == 1, -1, np.where(data == 0, 0, data - 1))
        # Scale the dataset to the new resolution
        # Resize geometry using interpolation
        scale_factor = 0.01 / 0.005  # If original resolution is 1
        # print(data.shape)
        data = zoom(data, (scale_factor, scale_factor, scale_factor), order=0)
        # print(data.shape)
        geoname = "./Input_3D/geometry_processed.h5"
        # Save the transformed data back to the file
        with h5py.File(geoname, 'w') as f:
            f.create_dataset('data', data=data)
            f.attrs['dx_dy_dz'] = (0.005, 0.005, 0.005)
            f.close()
    def run_base(self):

        # Run gprMax
        self.input = './Input_3D/Base{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.04

        sharp_domain =  self.square_size, 1, 1
        domain_2d = [
            float(sharp_domain[0] + 2 * pml + src_to_pml + 0.2), 
            float(sharp_domain[1] + 2 * pml + 0.2), 
            float(sharp_domain[2] + 2 * pml + src_to_pml + 0.2), 
        ]

        # Preprocess geometry

        try:
            with open('{}materials.txt'.format('Base_'), "w") as file:
                file.write('#material: {} 0 1 0 wall\n'.format(self.wall_permittivity))
                # for i in range(len(self.object_permittivity)):
                    # file.write('#material: {} 0 1 0 Object{}\n'.format(self.object_permittivity[i],i))
            self.preprocess(self.basefile)
        except Exception as e:
            print(e)

        src_position = [pml + src_to_pml + 0.2, 
                        0.5 + 0.1,  
                        pml + src_to_pml + 0.1]
        rx_position = [pml + src_to_pml + 0.2 + self.src_to_rx, 
                       0.5 + 0.1, 
                       pml + src_to_pml + 0.1]        
        
        src_steps = [(self.square_size-0.2)/ self.num_scan, 0, 0]
#         # print(src_steps)
        config = f'''

#title: Wall Object Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}

Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave

#hertzian_dipole: y {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml + src_to_pml + 0.1:.3f} {pml + 0.1:.3f} {pml+ src_to_pml + 0.2:.3f} ./Input_3D/geometry_processed.h5 Base_materials.txt
geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Base 
geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 Base n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                n=self.num_scan - self.restart + 1, 
                gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                n=self.num_scan - self.restart + 1, 
                # gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        
        try:
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ey'][()]
                dt = f1.attrs['dt']
                f1.close()

            rxnumber = 1
            rxcomponent = 'Ey'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            fig_width = 15
            fig_height = 15

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding


            os.rename(output_file, f'./Output_3D/Base/Base{self.i}.out')
            plt.savefig(f'./BaseImg_3D/Base{self.i}' + ".png")
        except Exception as e:
            print(e)


    def run_2D(self):

        # Run gprMax
        self.input = './Input_3D/Object{}.in'.format(self.i)
        pml_cells = 20
        pml = self.resol * pml_cells
        src_to_pml = 0.04

        sharp_domain =  self.square_size, 1, 1
        domain_2d = [
            float(sharp_domain[0] + 2 * pml + src_to_pml + 0.2), 
            float(sharp_domain[1] + 2 * pml + 0.2), 
            float(sharp_domain[2] + 2 * pml + src_to_pml + 0.2), 
        ]

        # Preprocess geometrys

        try:
            with open('{}materials.txt'.format('Obj_'), "w") as file:
                file.write('#material: {} 0 1 0 wall\n'.format(self.wall_permittivity))
                for i in range(len(self.object_permittivity)):
                    file.write('#material: {} 0 1 0 Object{}\n'.format(self.object_permittivity[i],i))
            self.preprocess(self.geofile)
        except Exception as e:
            print(e)

        src_position = [pml + src_to_pml + 0.2, 
                        0.5 + 0.1,  
                        pml + src_to_pml + 0.1]
        rx_position = [pml + src_to_pml + 0.2 + self.src_to_rx, 
                       0.5 + 0.1, 
                       pml + src_to_pml + 0.1]        
        
        src_steps = [(self.square_size-0.2)/ self.num_scan, 0, 0]
        config = f'''

#title: Wall Object Imaging

Configuration
#domain: {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f}
#dx_dy_dz: 0.005 0.005 0.005
#time_window: {self.time_window}

#pml_cells: {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells} {pml_cells}

Source - Receiver - Waveform
#waveform: ricker 1 1e9 my_wave

#hertzian_dipole: y {src_position[0]:.3f} {src_position[1]:.3f} {src_position[2]:.3f} my_wave 
#rx: {rx_position[0]:.3f} {rx_position[1]:.3f} {rx_position[2]:.3f}
#src_steps: {src_steps[0]:.3f} 0 0
#rx_steps: {src_steps[0]:.3f} 0 0

Geometry objects read

#geometry_objects_read: {pml + src_to_pml + 0.1:.3f} {pml + 0.1:.3f} {pml+ src_to_pml + 0.2:.3f}  ./Input_3D/geometry_processed.h5 Obj_materials.txt
geometry_objects_write: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} Object 
geometry_view: 0 0 0 {domain_2d[0]:.3f} {domain_2d[1]:.3f} {domain_2d[2]:.3f} 0.005 0.005 0.005 Object n

        '''

        with open(self.input, 'w') as f:
            f.write(config)
            f.close()
        try:
            api(self.input, 
                n=self.num_scan - self.restart + 1, 
                gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        except Exception as e:
                api(self.input, 
                n=self.num_scan - self.restart + 1, 
                # gpu=[0], 
                restart=self.restart,
                geometry_only=False, geometry_fixed=False)
        try:
        
            merge_files(str(self.input.replace('.in','')), True)
            output_file =str(self.input.replace('.in',''))+ '_merged.out'
            uncleaned_output_file = f'./Output_3D/WallObj/Wall_Obj{self.i}.out'
            dt = 0

            with h5py.File(output_file, 'r') as f1:
                data1 = f1['rxs']['rx1']['Ey'][()]
                dt = f1.attrs['dt']
                f1.close()

            # with h5py.File(f'./Output_3D/Base/Base{self.i}.out', 'r') as f1:
            #     data_source = f1['rxs']['rx1']['Ey'][()]

            with h5py.File(uncleaned_output_file, 'w') as f_out:
                f_out.attrs['dt'] = dt  # Set the time step attribute
                f_out.create_dataset('rxs/rx1/Ey', data=data1)

            rxnumber = 1
            rxcomponent = 'Ey'
            plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            fig_width = 15
            fig_height = 15
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plt.imshow(data1, cmap='gray', aspect='auto')
            plt.axis('off')
            ax.margins(0, 0)  # Remove any extra margins or padding
            fig.tight_layout(pad=0)  # Remove any extra padding
            plt.savefig(f'./WallObj_3D/Wall_Obj{self.i}' + ".png")

            # data1 = np.subtract(data1, data_source)

            # with h5py.File(output_file, 'w') as f_out:
            #     f_out.attrs['dt'] = dt  # Set the time step attribute
            #     f_out.create_dataset('rxs/rx1/Ey', data=data1)

            # # Draw data with normal plot
            # rxnumber = 1
            # rxcomponent = 'Ey'
            # plt = mpl_plot_Bscan("merged_output_data", data1, dt, rxnumber,rxcomponent)
            
            # fig_width = 15
            # fig_height = 15

            # fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # plt.imshow(data1, cmap='gray', aspect='auto')
            # plt.axis('off')
            # ax.margins(0, 0)  # Remove any extra margins or padding
            # fig.tight_layout(pad=0)  # Remove any extra padding

            # os.rename(output_file, f'./Output_3D/Object/Obj{self.i}.out')
            # plt.savefig(f'./ObjImg_3D/Obj{self.i}' + ".png")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wall Scanning for Through Wall Imaging")      
    parser.add_argument('--start', type=int, default=0, help='Start of the generated geometry')
    parser.add_argument('--end', type=int, default=5, help='End of the generated geometry')
    # data = np.load('SL_Obj3Dall_0_699.npz', allow_pickle=True)
    # data = np.load('SL_Obj3Dall_700_1500.npz', allow_pickle=True)
    data = np.load('./Geometry_3D/params_0_50.npz', allow_pickle=True)
    args = parser.parse_args()
    data_index = 0
    for i in range(args.start, args.end - args.start):
        i = i + data_index
        args.square_size = data['params'][i]['cube_size']/100
        args.wall_thickness = data['params'][i]['wall_thickness']/100
        # args.obj_width = data['params'][i]['rect_width']/100
        # args.obj_height = data['params'][i]['rect_height']/100
        args.wall_permittivity = round(data['params'][i]['permittivity_wall'], 2)
        args.object_permittivity = [round(p, 2) for p in data['params'][i]['permittivity_object']]
    # start  adaptor
        args.i = i
        wallimg = Wall_Func(args=args)
        print(args)
        # wallimg.view_geometry()
        wallimg.run_base()
        wallimg.run_2D()
