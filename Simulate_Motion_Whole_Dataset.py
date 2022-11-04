import glob
import os.path
import random
import numpy as np
from Utils import load_all_echoes, SimulateMotionForEachReadout, ImportMotionDatafMRI, transform_sphere
import nibabel as nib
import matplotlib.pyplot as plt
import time
import h5py
import argparse
import yaml
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#ff506e', '#69005f', 'tab:gray', 'tab:olive', 'tab:blue', 'peru'])


parser = argparse.ArgumentParser(description='Motion simulation')
parser.add_argument('--config_path', type=str, default='config/config_run_all.yaml', metavar='C',
                    help='path to configuration yaml file')
args = parser.parse_args()
with open(args.config_path, 'r') as stream_file:
    config_file = yaml.load(stream_file, Loader=yaml.FullLoader)

total = len(config_file['subject'])
if isinstance(config_file['simulation_nr'], list):
    total = len(config_file['subject']) * len(config_file['simulation_nr'])

current = 1
for sub, task in zip(config_file['subject'], config_file['task']):

    in_folder = config_file['in_folder']+config_file['dataset']+'/input/'+sub+'/t2star/'
    out_folder = config_file['out_folder']+config_file['dataset']+'/'+sub+'/'
    code_folder = config_file['code_folder']
    motion_folder = config_file['motion_folder']
    brainmask_file = config_file['in_folder']+config_file['dataset']+'/output/'+sub+'/'+task+'/qBOLD/T1w_coreg/rcBrMsk_CSF.nii'
    file_tag = task+'_acq-fullres_T2star.nii.gz'
    thr = config_file['motion_thr']
    save_magn_nifti = config_file['save_magn_nifti']
    include_transform = config_file['include_transform']
    include_inhomog = config_file['include_inhomog']
    if 'motion_from_h5' in config_file:
        motion_from_h5 = config_file['motion_from_h5']
        tag_h5 = config_file['tag_h5']
    else:
        motion_from_h5 = False
        tag_h5 = ''
    if 'motion_level' in config_file:
        motion_level = config_file['motion_level']
        check_motion = True
    else:
        check_motion = False

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print('File: ', in_folder, '**', file_tag)
    print('Results will be saved under: ', out_folder)

    print('############################\nStarting Simulation\n############################')
    if not motion_from_h5:
        # pick random motion files:
        if isinstance(config_file['simulation_nr'], list):
            simulation_nrs = config_file['simulation_nr']
            motion_files = [random.choice(glob.glob(motion_folder + '**.txt'))
                            for i in simulation_nrs]
        else:
            simulation_nrs = [config_file['simulation_nr']]
            motion_files = [random.choice(glob.glob(motion_folder + '**.txt'))]
    else:
        if isinstance(config_file['simulation_nr'], list):
            simulation_nrs = config_file['simulation_nr']
            motion_files = [glob.glob(out_folder + '**' + tag_h5 + '_' +
                                      str(s) + '**.h5')[0]
                            for s in simulation_nrs]
        else:
            motion_files = sorted(glob.glob(out_folder + '**' + tag_h5 +
                                            '**.h5'))
            simulation_nr = config_file['simulation_nr']
            simulation_nrs = np.arange(simulation_nr,
                                       len(motion_files)+simulation_nr)


    dset, affine = load_all_echoes(in_folder, file_tag)
    brainmask = np.rollaxis(nib.load(brainmask_file).get_fdata(), 2, 0)

    nr_slices = int(dset.shape[1])
    path_scan_order = code_folder + 'Scan_order_'+str(nr_slices)+'.txt'
    scan_length = int(np.loadtxt(path_scan_order)[-1, 0]) + 1  # duration of a scan in seconds

    for motion_file, simulation_nr in zip(motion_files, simulation_nrs):
        start = time.time()
        if include_transform:
            if include_inhomog:
                tag = 'sim_both_' + str(simulation_nr)
            else:
                tag = 'sim_rigid_' + str(simulation_nr)
        else:
            if include_inhomog:
                tag = 'sim_b0_' + str(simulation_nr)
            else:
                print('Please set either rigid transformations or B0-inhomogeneities (or both) to True.')
                break
        print('The result is saved with tag: ', tag)

        if os.path.exists(out_folder + sub + '_' + task + '_acq-fullres_T2star_' + tag + '.h5'):
            print('ERROR: Output file already exits. Please check and maybe change file naming, i.e. simulation_nr.')
            break

        # import motion data
        if not motion_from_h5:
            if check_motion:
                av_magn = -1
                while not motion_level[1] >= av_magn > motion_level[0]:
                    # import motion data
                    motion_file = random.choice(glob.glob(motion_folder + '**.txt'))
                    MotionImport = ImportMotionDatafMRI(filename=motion_file, scan_length=scan_length,
                                                        random_start_time=True)
                    times, T, R = MotionImport.get_motion_data(dset.shape, augment=True)

                    motion = np.array([times, T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T
                    motion_data = np.array([T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T

                    # look at average displacement of a sphere with radius 64mm
                    centroids, tr_coords = transform_sphere([12, 35, 112, 112], motion_data,
                                                            pixel_spacing=[3.3, 2, 2], radius=64)
                    # calculate reference through median
                    ind_median_centroid = np.argmin(
                        np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))
                    # calculate average voxel displacement magnitude
                    displ = tr_coords - tr_coords[ind_median_centroid]
                    magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)
                    av_magn = np.mean(magn)
                    print('Average Magntitude:', av_magn)

            else:
                MotionImport = ImportMotionDatafMRI(filename=motion_file, scan_length=scan_length,
                                                    random_start_time=True)
                times, T, R = MotionImport.get_motion_data(dset.shape, augment=True)
                # resort them to match dimensions of data:
                motion = np.array([times, T[:, 2], T[:, 1], T[:, 0], R[:, 2], R[:, 1], R[:, 0]]).T

        else:
            motion = h5py.File(motion_file, "r")['Motion_Curve']

        print('Motion scenario: ', motion_file)
        Simulation = SimulateMotionForEachReadout(motion, nr_pe_steps=92, brainmask=brainmask, path_scan_order=path_scan_order, motion_thr=thr,
                                                  include_transform=include_transform, include_inhomog=include_inhomog)

        magn, mask, full_mask = Simulation.create_mask_from_motion(dset)

        # Simulate motion:
        dset_sim = Simulation.simulate_all(dset)

        # save the simulated image:
        with h5py.File(out_folder+sub+'_'+task+'_acq-fullres_T2star_'+tag+'.h5', 'w') as h5_file:
            dset_1 = h5_file.create_dataset(name='Simulated_Data', shape=np.shape(dset_sim), data=dset_sim)
            dset_2 = h5_file.create_dataset(name='Motion_Curve', shape=np.shape(motion), data=motion)
            dset_2.attrs['Order_of_data'] = 'Times, z-translation, y-translation, x-translation, z-rotation, y-rotation, x-rotation'
            dset_3 = h5_file.create_dataset(name='Corruption_Mask', shape=np.shape(mask), data=mask)
            dset_3.attrs['Reference_method'] = 'Median'
            dset_3.attrs['Threshold'] = thr
            dset_3.attrs['Labels'] = '0 for motion > threshold, 1 for motion <= threshold (lines that can be included)'
            dset_4 = h5_file.create_dataset(name='Affine_Nifti_Transform', shape=np.shape(affine), data=affine)

        # optionally: save the simulated data as nifti files:
        if save_magn_nifti:
            for i in range(dset_sim.shape[0]):
                sim_echo = np.rollaxis(dset_sim[i], 0, 3)
                sim_nii = nib.Nifti1Image(abs(sim_echo), affine)
                nib.save(sim_nii, out_folder+sub+'_'+task+'_acq-fullres_T2star_'+tag + '_magn/echo_' + str(i) + '.nii')

        end = time.time()
        print('############################\n', str(current), ' out of ', str(total), ' done')
        print('############################\nSimulation took: ', (end - start) / 60, ' minutes.\n############################')
        current += 1
