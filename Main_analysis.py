import shutil
import numpy as np
import glob
import nibabel as nib
import os
import h5py
import subprocess
from skimage.metrics import structural_similarity
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#ff506e', '#69005f', 'tab:gray', '#0065bd', 'tab:olive', 'peru'])
from Utils import transform_sphere
from Utils_plot import show_stars, add_label


'''Simulations (rigid body transformations) performed with Simulate_Motion_Whole_Dataset.py for three different 
motion levels - e.g. with yaml file config/config_run_rigid_mild.
Also perform simulations with B0 inhomogeneities for same motion data with e.g. yaml file config/config_run_rigid_B0_mild
'''

# Define folders etc:
sim_folder = "/home/iml/hannah.eichhorn/Data/YoungHealthyVol/simulated_abstract/"
res_folder = "/home/iml/hannah.eichhorn/Results/Motion_simulation/Abstract_results/"
proc_folder = "/home/iml/hannah.eichhorn/Data/YoungHealthyVol/processed/"
sim_proc_folder = "/home/iml/hannah.eichhorn/Data/YoungHealthyVol/processed_simulated_abstr/"

datasets = ['DATA_Christine', 'DATA_Epp_4_task']
tasks = ['AIR', 'rest']
tags = ['rigid', 'both']


''' Look at motion for simulated images:'''
plot_sim_motion = False
if plot_sim_motion:
    motion_levels = [0.66, 1.33, 2.0]

    for tag in tags:
        av_displ = {'<' + str(motion_levels[0]): [], str(motion_levels[0]) + '-' + str(motion_levels[1]): [],
                    '>' + str(motion_levels[1]): []}
        sorted_files = {'<' + str(motion_levels[0]): [], str(motion_levels[0]) + '-' + str(motion_levels[1]): [],
                        '>' + str(motion_levels[1]): []}
        for dataset in datasets:
            for filename in glob.glob(sim_folder+dataset+"/**/**"+tag+"**.h5"):
                # load motion data
                tmp = h5py.File(filename, "r")['Motion_Curve']
                times, motion_data = tmp[:, 0], tmp[:, 1:]

                # look at average displacement of a sphere with radius 64mm
                centroids, tr_coords = transform_sphere([12, 35, 112, 112], motion_data,
                                                        pixel_spacing=[3.3, 2, 2], radius=64)
                # calculate reference through median
                ind_median_centroid = np.argmin(np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2, axis=1)))
                # calculate average voxel displacement magnitude
                displ = tr_coords - tr_coords[ind_median_centroid]
                magn = np.sqrt(displ[:, :, 0] ** 2 + displ[:, :, 1] ** 2 + displ[:, :, 2] ** 2)

                if np.mean(magn) < motion_levels[0]:
                    av_displ['<'+str(motion_levels[0])].append([np.mean(magn), dataset+'_'+os.path.basename(filename)])
                    #sorted_files['<1'].append(dataset+'_'+os.path.basename(filename))
                    save_name = 'mild_'+dataset+'_'+os.path.basename(filename)
                elif np.mean(magn) >= motion_levels[1]:
                    av_displ['>'+str(motion_levels[1])].append([np.mean(magn), dataset+'_'+os.path.basename(filename)])
                    #sorted_files['>2'].append(dataset+'_'+os.path.basename(filename))
                    save_name = 'strong_'+dataset+'_'+os.path.basename(filename)
                else:
                    av_displ[str(motion_levels[0])+'-'+str(motion_levels[1])].append([np.mean(magn), dataset+'_'+os.path.basename(filename)])
                    #sorted_files['1-2'].append(dataset+'_'+os.path.basename(filename))
                    save_name = 'moderate_'+dataset+'_'+os.path.basename(filename)

        for key, val in av_displ.items():
            matching = {'<'+str(motion_levels[0]): 'mild', str(motion_levels[0])+'-'+str(motion_levels[1]): 'moderate', '>'+str(motion_levels[1]): 'strong'}
            np.savetxt(res_folder+'Average_displacements_'+matching[key]+'_'+tag+'.txt', val, fmt='%s')

        # plot maximum motion parameters:
        plt.figure(figsize=(10, 7))
        plt.scatter(np.ones(len(av_displ['<'+str(motion_levels[0])])), np.array(av_displ['<'+str(motion_levels[0])])[:,0].astype(float))
        plt.scatter(np.ones(len(av_displ[str(motion_levels[0])+'-'+str(motion_levels[1])])) + 1, np.array(av_displ[str(motion_levels[0])+'-'+str(motion_levels[1])])[:,0].astype(float))
        plt.scatter(np.ones(len(av_displ['>'+str(motion_levels[1])])) + 2, np.array(av_displ['>'+str(motion_levels[1])])[:,0].astype(float))
        plt.xticks([1, 2, 3], ['<'+str(motion_levels[0]), str(motion_levels[0])+'-'+str(motion_levels[1]), '>'+str(motion_levels[1])])
        plt.xlabel('Motion levels')
        plt.ylabel('Average displacement')
        plt.savefig(res_folder+'Average_displacements_'+tag+'.png')
        plt.show()


''' Collect the simulated images in correct input folders for processing with MATLAB pipeline:'''
# FOR MULTIPLE TASKS; T2 NEEDS TO BE RENAMED TO MATCH FIRST CONDITION!
collect_input = False
if collect_input:
    for tag in tags:
        for dataset, task in zip(datasets, tasks):
            subjects = glob.glob(sim_folder+dataset+'/sub**')
            subjects = [os.path.basename(s) for s in subjects]

            for level in ['mild', 'moderate', 'strong']:
                tmp, files = np.loadtxt(res_folder+'Average_displacements_'+level+'_'+tag+'.txt', unpack=True, dtype=str)
                files = [f.replace(dataset+'_', '') for f in files if dataset in f]

                for f in files:
                    # find subject:
                    subject = ''
                    for sub in subjects:
                        if sub in f:
                            subject = sub
                    if subject == '':
                        print('No subject found!')

                    # find the simulation_nr:
                    sim_nr = f[-4:-3]

                    new_subject = subject+'-'+level+'-'+tag+'-'+sim_nr

                    if os.path.exists(sim_proc_folder+dataset+'/input/'+new_subject+'/'):
                        print('Subject', new_subject, 'already exists in', sim_proc_folder+dataset)
                        print('This subject is skipped.')
                        break

                    # copy input folder from processed folder
                    print('copied', proc_folder+dataset+'/input/'+subject+'/', 'to',
                          sim_proc_folder+dataset+'/input/'+new_subject+'/')
                    shutil.copytree(proc_folder+dataset+'/input/'+subject+'/',
                                    sim_proc_folder+dataset+'/input/'+new_subject+'/')

                    # copy part of the output folders as well (so that anatomy processing does not have to be repeated)
                    print('copied', proc_folder + dataset + '/output/' + subject + '/FLAIR/', 'to',
                          sim_proc_folder + dataset + '/output/' + new_subject + '/FLAIR/')
                    shutil.copytree(proc_folder + dataset + '/output/' + subject + '/FLAIR/',
                                    sim_proc_folder + dataset + '/output/' + new_subject + '/FLAIR/')
                    print('copied', proc_folder + dataset + '/output/' + subject + '/T1w/', 'to',
                          sim_proc_folder + dataset + '/output/' + new_subject + '/T1w/')
                    shutil.copytree(proc_folder + dataset + '/output/' + subject + '/T1w/',
                                    sim_proc_folder + dataset + '/output/' + new_subject + '/T1w/')


                    # rename all subjects into subject-level-tag:
                    sub_files = [s for s in
                                 glob.glob(sim_proc_folder+dataset+'/input/'+new_subject+'/**', recursive=True)
                                 if os.path.isfile(s)]
                    for s in sub_files:
                        subst = os.path.join(os.path.split(s)[0], os.path.split(s)[1].replace(subject, new_subject))
                        os.rename(s, subst)

                    sub_files = [s for s in
                                 glob.glob(sim_proc_folder + dataset + '/output/' + new_subject + '/**', recursive=True)
                                 if os.path.isfile(s)]
                    for s in sub_files:
                        subst = os.path.join(os.path.split(s)[0], os.path.split(s)[1].replace(subject, new_subject))
                        os.rename(s, subst)

                    # delete unnecessary files from T2 Star directory (FR nii):
                    t2star_delete = glob.glob(sim_proc_folder+dataset+'/input/'+new_subject+'/t2star/**task-'+task+'**fullres**.nii**')
                    for nr, t in enumerate(t2star_delete):
                        if nr == 0:
                            hd = nib.load(t).header
                        os.remove(t)

                    # fill with nii files from simulation:
                    sim = h5py.File(sim_folder+dataset+'/'+subject+'/'+f, "r")['Simulated_Data']
                    affine = h5py.File(sim_folder+dataset+'/'+subject+'/'+f, "r")['Affine_Nifti_Transform']

                    for i in range(sim.shape[0]):
                        sim_echo = np.rollaxis(sim[i], 0, 3)

                        # magn:
                        sim_nii = nib.Nifti1Image(abs(sim_echo), affine, hd)
                        if i < 9:
                            save_name = new_subject+'_echo-0'+str(i+1)+'_task-'+task+'_acq-fullres_T2star.nii.gz'
                        else:
                            save_name = new_subject+'_echo-' +str(i+1)+ '_task-'+task+'_acq-fullres_T2star.nii.gz'
                        nib.save(sim_nii, sim_proc_folder+dataset+'/input/'+new_subject+'/t2star/' + save_name)

                        # real:
                        sim_nii = nib.Nifti1Image(np.real(sim_echo), affine, hd)
                        if i < 9:
                            save_name = new_subject + '_echo-0' + str(i + 1) + '_real_task-'+task+'_acq-fullres_T2star.nii.gz'
                        else:
                            save_name = new_subject + '_echo-' + str(i + 1) + '_real_task-'+task+'_acq-fullres_T2star.nii.gz'
                        nib.save(sim_nii, sim_proc_folder+dataset+'/input/' + new_subject + '/t2star/' + save_name)

                        # imag:
                        sim_nii = nib.Nifti1Image(np.imag(sim_echo), affine, hd)
                        if i < 9:
                            save_name = new_subject + '_echo-0' + str(i + 1) + '_imaginary_task-'+task+'_acq-fullres_T2star.nii.gz'
                        else:
                            save_name = new_subject + '_echo-' + str(i + 1) + '_imaginary_task-'+task+'_acq-fullres_T2star.nii.gz'
                        nib.save(sim_nii, sim_proc_folder+dataset+'/input/' + new_subject + '/t2star/' + save_name)

                    # delete other tasks from T2star data:
                    t2star_delete = glob.glob(sim_proc_folder + dataset + '/input/' + new_subject + '/t2star/****')
                    t2star_delete = [t for t in t2star_delete if 'task-'+task not in t]
                    for t in t2star_delete:
                        os.remove(t)


''' For T2* mapping and calculation of R2' maps: process the above created folder sim_proc_folder with Matlab pipeline from:
Kaczmarz, Stephan, Fahmeed Hyder, and Christine Preibisch. “Oxygen Extraction Fraction Mapping with Multi-Parametric 
Quantitative BOLD MRI: Reduced Transverse Relaxation Bias Using 3D-GraSE Imaging.” NeuroImage 220 (October 2020): 117095.
'''


'''Register the ground truth T2* data to the simulated data:'''
register = False
if register:
    for tag in tags:
        for dataset, task in zip(datasets, tasks):
            subjects = glob.glob(sim_folder+dataset+'/sub**')
            subjects = [os.path.basename(s) for s in subjects]

            for subject in subjects:
                for level, sim_nrs in zip(['mild', 'moderate', 'strong'], [[0,3,6], [1,4,7], [2,5,8]]):
                    for sim_nr in sim_nrs:

                        new_subject = subject + '-' + level + '-' + tag + '-' + str(sim_nr)

                        source_dir = sim_proc_folder + 'coreg_orig/' + dataset + '/' + new_subject + '/task-' + task + '/orig_T2s/'
                        out_dir = sim_proc_folder + 'coreg_orig/' + dataset + '/' + new_subject + '/task-' + task + '/orig_T2s_coreg/'
                        target_dir = sim_proc_folder + dataset + '/output/' + new_subject + '/task-' + task + '/qBOLD/T2S/FR/Magn/'

                        if os.path.exists(source_dir):
                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            print('ERROR: Source directory already exists, please look into this!')
                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            break

                        # copy original files to source_dir:
                        print('copied', proc_folder + dataset + '/output/' + subject + '/task-' + task + '/qBOLD/T2S/FR/Magn/', 'to',
                              source_dir)
                        shutil.copytree(proc_folder + dataset + '/output/' + subject + '/task-' + task + '/qBOLD/T2S/FR/Magn/',
                                        source_dir)
                        file = glob.glob(proc_folder + dataset + '/output/' + subject + '/task-' + task + '/qBOLD/T2S/T2s_uncorr**.nii')[0]
                        print('copied', file, 'to', source_dir+os.path.basename(file))
                        shutil.copyfile(file, source_dir+os.path.basename(file))
                        file = glob.glob(
                            proc_folder + dataset + '/output/' + subject + '/task-' + task + '/qBOLD/sR2strich_uncorr**.nii')[
                            0]
                        print('copied', file, 'to', source_dir + os.path.basename(file))
                        shutil.copyfile(file, source_dir + os.path.basename(file))

                        subprocess.run('matlab -r "Coregister_GT_T2star ' + subject + ' '+ new_subject+ ' ' + source_dir + ' ' + target_dir + ' ' + out_dir+';exit"',
                                       shell=True)




'''Analyse motion-induced errors in images and derived parameter maps:'''
analyse_output = False
if analyse_output:
    # 1) analysis of T2*-weighted data - single echoes:
    SSIM = {}
    for tag in tags:
        for level, sim_nrs in zip(['mild', 'moderate', 'strong'], [[0, 3, 6], [1, 4, 7], [2, 5, 8]]):
            for sim_nr in sim_nrs:
                for echo_ in range(1, 13):
                    SSIM[level + '_' + tag + '_' + str(echo_)] = []
                    if echo_ < 10:
                        echo = '0' + str(echo_)
                    else:
                        echo = str(echo_)

                    for dataset, task in zip(datasets, tasks):
                        subjects = sorted(glob.glob(sim_folder + dataset + '/sub**'))
                        subjects = [os.path.basename(s) for s in subjects]

                        for subject in subjects:
                            new_subject = subject + '-' + level + '-' + tag + '-' + str(sim_nr)
                            # filenames:
                            f_T2s_sim = glob.glob(sim_proc_folder + dataset + '/input/' + new_subject + '/t2star/**echo-' +
                                                  echo + '_task-' + task + '_acq-fullres**.nii**')[0]
                            f_T2s_gt = glob.glob(
                                sim_proc_folder + 'coreg_orig/' + dataset + '/' + new_subject + '/task-' + task + '/orig_T2s_coreg/**echo-' +
                                echo + '_task-' + task + '_acq-fullres**.nii**')[0]

                            # load nii files:
                            T2s_sim = nib.load(f_T2s_sim).get_fdata()
                            T2s_gt = nib.load(f_T2s_gt).get_fdata()

                            # slice-wise ssim:
                            ssim = []
                            for i in range(0, np.shape(T2s_sim)[2]):
                                peak = np.amax(T2s_gt[:, :, i])
                                ssim.append(structural_similarity(T2s_gt[:, :, i], T2s_sim[:, :, i], data_range=peak,
                                                                  gaussian_weights=True))

                            SSIM[level + '_' + tag + '_' + str(echo_)].append(np.mean(ssim))

    # perform Wilcoxon signed rank tests and multiple comparison correction:
    p_values, descr = [], []
    # between rigid and both:
    for level in ['mild', 'moderate', 'strong']:
        for echo_ in range(1, 13):
            tmp, p = wilcoxon(SSIM[level + '_rigid_' + str(echo_)],
                              SSIM[level + '_both_' + str(echo_)],
                              alternative='two-sided')
            p_values.append(p)
            descr.append(level+'_rigid-both_'+str(echo_))

    p_values, descr = np.array(p_values), np.array(descr)

    rej, p_values_cor, _, __ = multipletests(p_values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)


    # plot the results for certain echoes:
    plot_echoes = [1, 6, 12]
    plt.figure(figsize=(13, 4))
    for nr, p in enumerate(plot_echoes):
        plot_a = [SSIM['mild_rigid_' + str(p)], SSIM['moderate_rigid_' + str(p)], SSIM['strong_rigid_' + str(p)]]
        plot_b = [SSIM['mild_both_' + str(p)], SSIM['moderate_both_' + str(p)], SSIM['strong_both_' + str(p)]]

        plt.subplot(int(len(plot_echoes) / 3), 3, nr + 1)
        labels = []
        add_label(plt.violinplot(plot_a, positions=[2.2, 4.2, 6.2], showmeans=True, showextrema=False), 'rigid',
                  labels=labels)
        add_label(plt.violinplot(plot_b, positions=[2.8, 4.8, 6.8], showmeans=True, showextrema=False), 'rigid + B0',
                  labels=labels)

        p_val = np.array([p_values_cor[descr==d] for d in descr if d.endswith('rigid-both_' + str(p))]).flatten()
        h = []
        for a, b in zip(plot_a, plot_b):
            h.append(np.amax(a))
            h.append(np.amax(b))
        show_stars(p_cor=np.array(p_val), ind=[[0, 1], [2, 3], [4, 5]], bars=[2.2, 2.8, 4.2, 4.8, 6.2, 6.8],
                   heights=h)

        plt.xticks([2.5, 4.5, 6.5], ['mild', 'moderate', 'strong'])
        plt.legend(*zip(*labels), loc='lower left')
        plt.ylabel('Mean SSIM over slices')
        plt.title('Echo ' + str(p))
        plt.ylim(0.4, 1.08)

    plt.tight_layout()
    plt.savefig(res_folder + 'SSIM_analysis.png', dpi=300)
    plt.show()


    # 2) analysis of parameter maps:
    T2s = {'gt': []}
    T2s_diff_GM, T2s_diff_bm = {}, {}
    R2p_diff_GM, R2p_diff_bm = {}, {}
    T2s_std = {'gt': []}
    R2p = {'gt': []}
    R2p_std = {'gt': []}

    gt = True
    for tag in tags:
        for level, sim_nrs in zip(['mild', 'moderate', 'strong'], [[0, 3, 6], [1, 4, 7], [2, 5, 8]]):
            T2s[level + '_' + tag] = []
            T2s_diff_GM[level + '_' + tag] = []
            T2s_diff_bm[level + '_' + tag] = []
            T2s_std[level + '_' + tag] = []
            R2p[level + '_' + tag] = []
            R2p_std[level + '_' + tag] = []
            R2p_diff_GM[level + '_' + tag] = []
            R2p_diff_bm[level + '_' + tag] = []
            for sim_nr in sim_nrs:
                for dataset, task in zip(datasets, tasks):
                    subjects = sorted(glob.glob(sim_folder + dataset + '/sub**'))
                    subjects = [os.path.basename(s) for s in subjects]

                    for subject in subjects:
                        new_subject = subject + '-' + level + '-' + tag + '-' + str(sim_nr)
                        # filenames:
                        f_T2 = proc_folder+dataset+'/output/'+subject+'/task-'+task+'/qBOLD/T2.nii'
                        f_T2s_sim = glob.glob(sim_proc_folder+dataset+'/output/'+new_subject+'/task-'+task+'/qBOLD/T2S/T2s_uncorr**.nii')[0]
                        f_T2s_gt = glob.glob(sim_proc_folder+'coreg_orig/'+dataset+'/'+new_subject+'/task-'+task+'/orig_T2s_coreg/rT2s_uncorr**.nii')[0]
                        f_GM_seg_sim = glob.glob(sim_proc_folder + dataset + '/output/' + new_subject + '/task-' + task + '/qBOLD/T1w_coreg/rc1sub**.nii')[0]
                        f_bm_seg_sim = sim_proc_folder + dataset + '/output/' + new_subject + '/task-' + task + '/qBOLD/T1w_coreg/rcBrMsk_CSF.nii'
                        f_R2p_sim = sim_proc_folder+dataset+'/output/'+new_subject+'/task-'+task+'/qBOLD/sR2strich_uncorr.nii'
                        f_R2p_gt = sim_proc_folder+'coreg_orig/'+dataset+'/'+new_subject+'/task-'+task+'/orig_T2s_coreg/rsR2strich_uncorr.nii'

                        # load nii files:
                        T2 = nib.load(f_T2).get_fdata()
                        T2s_sim = nib.load(f_T2s_sim).get_fdata()
                        T2s_gt = nib.load(f_T2s_gt).get_fdata()
                        R2p_sim = nib.load(f_R2p_sim).get_fdata()
                        R2p_gt = nib.load(f_R2p_gt).get_fdata()
                        GM_seg_sim = nib.load(f_GM_seg_sim).get_fdata()
                        bm_seg_sim = nib.load(f_bm_seg_sim).get_fdata()
                        # threshold the registered GM segmentation
                        GM_seg_sim[GM_seg_sim < 0.5] = 0.
                        GM_seg_sim[GM_seg_sim > 0.5] = 1.
                        bm_seg_sim[bm_seg_sim < 0.5] = 0.
                        bm_seg_sim[bm_seg_sim > 0.5] = 1.

                        # analysis in GM:
                        T2s_gt_GM = T2s_gt[GM_seg_sim == 1]
                        T2s_sim_GM = T2s_sim[GM_seg_sim == 1]
                        R2p_gt_GM = R2p_gt[GM_seg_sim == 1]
                        R2p_sim_GM = R2p_sim[GM_seg_sim == 1]

                        if gt:
                            T2s['gt'].append(np.nanmean(T2s_gt_GM))
                            T2s_std['gt'].append(np.nanstd(T2s_gt_GM))
                            R2p['gt'].append(np.nanmean(R2p_gt_GM))
                            R2p_std['gt'].append(np.nanstd(R2p_gt_GM))
                        T2s[level + '_' + tag].append(np.mean(T2s_sim_GM))
                        T2s_std[level + '_' + tag].append(np.std(T2s_sim_GM))
                        R2p[level + '_' + tag].append(np.mean(R2p_sim_GM))
                        R2p_std[level + '_' + tag].append(np.std(R2p_sim_GM))

                        T2s_diff_GM[level + '_' + tag].append(np.nanmean(abs(T2s_sim_GM-T2s_gt_GM)))
                        R2p_diff_GM[level + '_' + tag].append(np.nanmean(abs(R2p_sim_GM - R2p_gt_GM)))

                        # analysis in brainmask:
                        T2s_gt_bm = T2s_gt[bm_seg_sim == 1]
                        T2s_sim_bm = T2s_sim[bm_seg_sim == 1]
                        R2p_gt_bm = R2p_gt[bm_seg_sim == 1]
                        R2p_sim_bm = R2p_sim[bm_seg_sim == 1]

                        T2s_diff_bm[level + '_' + tag].append(np.nanmean(abs(T2s_sim_bm - T2s_gt_bm)))
                        R2p_diff_bm[level + '_' + tag].append(np.nanmean(abs(R2p_sim_bm - R2p_gt_bm)))

            gt = False


    # perform Wilcoxon signed rank tests and multiple comparison correction:
    p_values, descr = [], []
    # between rigid and both:
    for level in ['mild', 'moderate', 'strong']:
        tmp, p = wilcoxon(T2s_diff_GM[level + '_rigid'],
                          T2s_diff_GM[level + '_both'],
                          alternative='two-sided')
        p_values.append(p)
        descr.append(level + '_rigid-both_T2s')
    for level in ['mild', 'moderate', 'strong']:
        tmp, p = wilcoxon(R2p_diff_GM[level + '_rigid'],
                          R2p_diff_GM[level + '_both'],
                          alternative='two-sided')
        p_values.append(p)
        descr.append(level + '_rigid-both_R2p')

    p_values, descr = np.array(p_values), np.array(descr)
    rej, p_values_cor, _, __ = multipletests(p_values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

    # create violin plots for comparing the different scenarios (T2star difference):
    plot_1a = [T2s_diff_GM['mild_rigid'], T2s_diff_GM['moderate_rigid'], T2s_diff_GM['strong_rigid']]
    plot_1b = [T2s_diff_GM['mild_both'], T2s_diff_GM['moderate_both'], T2s_diff_GM['strong_both']]
    plot_2a = [R2p_diff_GM['mild_rigid'], R2p_diff_GM['moderate_rigid'], R2p_diff_GM['strong_rigid']]
    plot_2b = [R2p_diff_GM['mild_both'], R2p_diff_GM['moderate_both'], R2p_diff_GM['strong_both']]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    labels = []
    add_label(plt.violinplot(plot_1a, positions=[2.2, 4.2, 6.2], showmeans=True, showextrema=False), 'rigid',
              labels=labels)
    add_label(plt.violinplot(plot_1b, positions=[2.8, 4.8, 6.8], showmeans=True, showextrema=False), 'rigid + B0',
              labels=labels)

    p_val = np.array([p_values_cor[descr == d] for d in descr if d.endswith('_rigid-both_T2s')]).flatten()
    h = []
    for a, b in zip(plot_1a, plot_1b):
        h.append(np.amax(a))
        h.append(np.amax(b))
    show_stars(p_cor=np.array(p_val), ind=[[0, 1], [2, 3], [4, 5]], bars=[2.2, 2.8, 4.2, 4.8, 6.2, 6.8],
               heights=h)

    ax = plt.gca()

    def to_perc(x):
        return x / np.mean(T2s['gt']) * 100
    def inv_to_perc(x):
        return x * np.mean(T2s['gt']) / 100

    secax = ax.secondary_yaxis('right', functions=(to_perc, inv_to_perc))
    secax.set_ylabel('Percentage voxel error [%]', c='dimgrey')
    secax.tick_params(axis='y', colors='dimgrey')

    plt.xticks([2.5, 4.5, 6.5], ['mild', 'moderate', 'strong'])
    plt.legend(*zip(*labels), loc='upper left')
    plt.ylabel('Mean T2* voxel error [ms]')
    plt.title('T2* voxel error in gray matter')

    plt.subplot(1, 2, 2)
    labels = []
    add_label(plt.violinplot(plot_2a, positions=[2.2, 4.2, 6.2], showmeans=True, showextrema=False), 'rigid',
              labels=labels)
    add_label(plt.violinplot(plot_2b, positions=[2.8, 4.8, 6.8], showmeans=True, showextrema=False), 'rigid + B0',
              labels=labels)

    p_val = np.array([p_values_cor[descr == d] for d in descr if d.endswith('_rigid-both_R2p')]).flatten()
    h = []
    for a, b in zip(plot_2a, plot_2b):
        h.append(np.amax(a))
        h.append(np.amax(b))
    show_stars(p_cor=np.array(p_val), ind=[[0, 1], [2, 3], [4, 5]], bars=[2.2, 2.8, 4.2, 4.8, 6.2, 6.8],
               heights=h)

    ax = plt.gca()

    def to_perc(x):
        return x / np.mean(R2p['gt']) * 100
    def inv_to_perc(x):
        return x * np.mean(R2p['gt']) / 100

    secax = ax.secondary_yaxis('right', functions=(to_perc, inv_to_perc))
    secax.set_ylabel('Percentage voxel error [%]', c='dimgrey')
    secax.tick_params(axis='y', colors='dimgrey')

    plt.xticks([2.5, 4.5, 6.5], ['mild', 'moderate', 'strong'])
    plt.legend(*zip(*labels), loc='upper left')
    plt.ylabel('Mean R2\' voxel error [1/s]')
    plt.title('R2\' voxel error in gray matter')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(res_folder + 'T2s_R2p_voxel_error.png', dpi=300)
    plt.show()

    print('Mean T2* (GM) in ground truth scans: ', np.mean(T2s['gt']), '+-', np.std(T2s['gt']))
    print('Mean R2\' (GM) in ground truth scans: ', np.mean(R2p['gt']), '+-', np.std(R2p['gt']))
    print('##########')
    print('Mean T2* error (GM) in all mild simulations: ',
          np.mean([np.mean(T2s_diff_GM['mild_rigid']), np.mean(T2s_diff_GM['mild_both'])]))
    print('Percentage error: ',
          np.mean([np.mean(T2s_diff_GM['mild_rigid']), np.mean(T2s_diff_GM['mild_both'])]) / np.mean(T2s['gt']) * 100)
    print('Mean R2\' error (GM) in all mild simulations: ',
          np.mean([np.mean(R2p_diff_GM['mild_rigid']), np.mean(R2p_diff_GM['mild_both'])]))
    print('Percentage error: ',
          np.mean([np.mean(R2p_diff_GM['mild_rigid']), np.mean(R2p_diff_GM['mild_both'])]) / np.mean(R2p['gt']) * 100)


''' Plot example images '''
plot_example = False
if plot_example:
    subject = 'sub-p027'
    subject_real = 'sub-p048'
    task = 'calc'
    level = 'mild'
    sim_nr = 6
    slice = 19
    slice_real = 17

    plt.figure(figsize=(10,8))
    nr = 1

    for echo in ['01', '06', '12']:
        motion_free = glob.glob(proc_folder+'DATA_Epp_4_task/input/'+subject+'/t2star/**echo-'+echo+'_task-rest**fullres**nii**')[0]
        real_motion = glob.glob(proc_folder + 'DATA_Epp_4_task/input/' + subject_real + '/t2star/**echo-' + echo + '_task-'+task+'**fullres**nii**')[0]
        rigid = glob.glob(sim_proc_folder + 'DATA_Epp_4_task/input/'+subject+'-'+level+'-rigid'+'-'+str(sim_nr)+ '/t2star/**echo-' + echo+'_task-rest**fullres**nii**')[0]
        both = glob.glob(sim_proc_folder + 'DATA_Epp_4_task/input/' + subject + '-' + level + '-both'+'-'+str(sim_nr) + '/t2star/**echo-' + echo+'_task-rest**fullres**nii**')[0]

        motion_free = nib.load(motion_free).get_fdata()
        real_motion = nib.load(real_motion).get_fdata()
        rigid = nib.load(rigid).get_fdata()
        both = nib.load(both).get_fdata()

        plt.subplot(3, 4, nr)
        plt.imshow(motion_free[:, ::-1, slice].T, cmap='gray')
        plt.axis('off')
        if nr == 1:
            plt.title('Motion-free', fontsize=9)

        plt.subplot(3, 4, nr+1)
        plt.imshow(rigid[:, ::-1, slice].T, cmap='gray')
        plt.axis('off')
        if nr == 1:
            plt.title('Simulation with \nrigid transformations', fontsize=9)

        plt.subplot(3, 4, nr+2)
        plt.imshow(both[:, ::-1, slice].T, cmap='gray')
        plt.axis('off')
        if nr == 1:
            plt.title('Simulation with \nrigid transformations\nand B0 inhomogeneities', fontsize=9)

        plt.subplot(3, 4, nr+3)
        plt.imshow(real_motion[:, ::-1, slice_real].T, cmap='gray')
        plt.axis('off')
        if nr == 1:
            plt.title('Real motion\n(different subject)', fontsize=9)

        nr+=4

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.001)
    plt.savefig(res_folder + 'Example_Epp_'+subject+'_'+level+'_'+str(sim_nr)+'.png', dpi=300, bbox_inches='tight')
    plt.show()


    # different motion levels compared:
    tag = 'both'
    echo_1 = '01'
    echo_2 = '12'

    plt.figure(figsize=(14, 15))
    nr = 1

    for level, sim_nr in zip(['mild', 'moderate', 'strong'], [6, 1, 5]):
        sim_1 = glob.glob(sim_proc_folder + 'DATA_Epp_4_task/input/' + subject + '-' + level + '-'+tag+ '-' + str(
            sim_nr) + '/t2star/**echo-' + echo_1 + '_task-rest**fullres**nii**')[0]
        sim_2 = glob.glob(sim_proc_folder + 'DATA_Epp_4_task/input/' + subject + '-' + level + '-' + tag + '-' + str(
            sim_nr) + '/t2star/**echo-' + echo_2 + '_task-rest**fullres**nii**')[0]
        motion = glob.glob(sim_folder+'DATA_Epp_4_task/'+subject+'/**'+tag+ '_' + str(sim_nr)+'**.h5')[0]

        # load motion data
        tmp = h5py.File(motion, "r")['Motion_Curve']
        times, motion_data = tmp[:, 0], tmp[:, 1:]

        sim_1 = nib.load(sim_1).get_fdata()
        sim_2 = nib.load(sim_2).get_fdata()

        if nr == 1:
            ax1 = plt.subplot(6, 3, nr)
        else:
            plt.subplot(6, 3, nr, sharey=ax1)
        plt.title(level + ' motion', fontsize=20)
        plt.plot(times, motion_data[:, 0], label='T_x')
        plt.plot(times, motion_data[:, 1], label='T_y')
        plt.plot(times, motion_data[:, 2], label='T_z')

        if nr == 1:
            plt.ylabel('Translation [mm]', fontsize=11)
            plt.legend(loc='best', fontsize=8)
        else:
            plt.yticks(color='w')
        plt.xticks(color='w')

        if nr == 1:
            ax2 = plt.subplot(6, 3, nr+3)
        else:
            plt.subplot(6, 3, nr+3, sharey=ax2)
        plt.plot(times, motion_data[:, 3], label='R_x')
        plt.plot(times, motion_data[:, 4], label='R_y')
        plt.plot(times, motion_data[:, 5], label='R_z')
        if nr == 1:
            plt.legend(loc='best', fontsize=8)
            plt.ylabel('Rotation [°]', fontsize=11)
        else:
            plt.yticks(color='w')
        plt.xlabel('Time [s]', fontsize=11)

        plt.subplot(3, 3, nr + 3)
        plt.imshow(sim_1[:, ::-1, slice].T, cmap='gray')
        plt.axis('off')

        plt.subplot(3, 3, nr + 6)
        plt.imshow(sim_2[:, ::-1, slice].T, cmap='gray')
        plt.axis('off')

        nr += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.23)
    plt.savefig(res_folder + 'Example_diff_levels_Epp' + subject + '_' + level + '.png', dpi=300,
                bbox_inches='tight')
    plt.show()


    # for setup figure:
    motion = glob.glob(sim_folder + 'DATA_Epp_4_task/' + subject + '/**' + tag + '_' + str(6) + '**.h5')[0]

    # load motion data
    tmp = h5py.File(motion, "r")['Motion_Curve']
    times, motion_data = tmp[:, 0], tmp[:, 1:]

    plt.figure(figsize=(10,5))
    plt.subplot(2, 1, 1)
    plt.plot(times, motion_data[:, 0], label='T_x')
    plt.plot(times, motion_data[:, 1], label='T_y')
    plt.plot(times, motion_data[:, 2], label='T_z')
    plt.ylabel('Translation [mm]')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(times, motion_data[:, 3], label='R_x')
    plt.plot(times, motion_data[:, 4], label='R_y')
    plt.plot(times, motion_data[:, 5], label='R_z')
    plt.legend(loc='best')
    plt.xlabel('Time [s]')
    plt.ylabel('Rotation [°]')
    plt.tight_layout()
    plt.show()
