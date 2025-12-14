from ANT_data_processing.feature_analyse_and_classify.feature_visualization import get_filelist
from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.stats as stats
from einops import rearrange
from pycirclize import Circos
from matplotlib import cm
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.io as sio
import scipy
import os
from scipy.io import savemat
from scipy.stats import ttest_ind
from source_atlas_inter import get_atlas_net,get_atlas_connectivity_people_type_fre
from find_coord import get_position_400
from matplotlib.colors import to_rgba_array
from nilearn import plotting


def connectivity_altas_ttest_plot(dict_people_type=['ASD_H', 'TD_P']):
    dict_atlas = get_atlas_net()
    new_sequence = []
    positions = {}
    save_tp_mat = {}
    position = 0
    for i_net in dict_atlas.keys():
        values = dict_atlas[i_net]
        new_sequence.extend(values)
        positions[i_net] = list(range(position, position + len(values)))
        position += len(values)
    ticks = []
    labels = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
    for name, index_range in positions.items():
        middle = (index_range[0] + index_range[-1]) / 2
        ticks.append(middle)
    fre_band_list = ['jiang_gamma_alpha_max', 'jiang_gamma_theta_max',
                     'jiang_beta_theta_max', 'jiang_beta_alpha_max']
    dict_resting_type = ['ec_avg']
    path_root = r"D:\data\KLC\average\sout_atlas_pac_connectivity_mat"
    path_save_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_new"
    if not os.path.isdir(path_save_root):
        os.makedirs(path_save_root)
    for i_resting_type in dict_resting_type:
        path_mat_resting_type = os.path.join(path_root, i_resting_type)
        path_save_resting_type = os.path.join(path_save_root, i_resting_type)
        for fre_index, fre_band in enumerate(fre_band_list):
            figure_name = f'{dict_people_type[0]}__{dict_people_type[1]}_{fre_band}'
            path_save_fig_t = os.path.join(path_save_root, "t_"+figure_name)
            path_save_fig_p = os.path.join(path_save_root, "p_" + figure_name)
            connectivity_0 = get_atlas_connectivity_people_type_fre(path_mat_resting_type, dict_people_type[0],
                                                                    fre_band, new_sequence)
            connectivity_1 = get_atlas_connectivity_people_type_fre(path_mat_resting_type, dict_people_type[1],
                                                                    fre_band, new_sequence)
            t_values = np.zeros((400, 400))
            p_values = np.zeros((400, 400))
            for i in range(400):
                for j in range(400):
                    t_values[i, j], p_values[i, j] = stats.ttest_ind(connectivity_0[:, i, j], connectivity_1[:, i, j])
            save_tp_mat[ "t_"+ figure_name] = t_values
            save_tp_mat["p_" + figure_name] = p_values
            t_values_ = t_values.copy()
            p_values_ = p_values.copy()
            t_values_[p_values>=0.001] = np.nan
            p_values_[p_values>=0.001] = np.nan
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.figure(figsize=(12, 10))
            sns.set_context("paper", font_scale=2)
            ax = sns.heatmap(t_values_, cmap="RdBu_r", cbar=True, vmin=-2, vmax=2)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            cmap = ax.collections[0].cmap
            for name, index_range in positions.items():
                start = index_range[0]
                end = index_range[-1]
                ax.add_patch(
                    plt.Rectangle((start, start), end - start, end - start, fill=True,facecolor='white',edgecolor='black', lw=3))
                # connectivity_net_name_0 = np.mean(connectivity_0[:, index_range, index_range], axis=1)
                # connectivity_net_name_1 = np.mean(connectivity_1[:, index_range, index_range], axis=1)
            plt.title("t_"+figure_name)
            plt.savefig(path_save_fig_t+ ".png", bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(12, 10))
            sns.set_context("paper", font_scale=2)
            ax = sns.heatmap(p_values_, cmap = "Oranges_r", cbar=True)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            cmap = ax.collections[0].cmap
            for name, index_range in positions.items():
                start = index_range[0]
                end = index_range[-1]
                ax.add_patch(plt.Rectangle((start, start), end - start, end - start, fill=True, facecolor='white',edgecolor='black', lw=3))
                # connectivity_net_name_0 = np.mean(connectivity_0[:, index_range, index_range], axis=1)
                # connectivity_net_name_1 = np.mean(connectivity_1[:, index_range, index_range], axis=1)
                # t_value, p_value = stats.ttest_ind(connectivity_net_name_0, connectivity_net_name_1)
                # color = cmap((t_value - (-2)) / (2 - (-2)))  # 根据 t_value 在 color map 上的位置选择颜色
                # ax.plot([start, end], [start, end], color=color, lw=3)
            plt.title("p_" + figure_name)
            plt.savefig(path_save_fig_p+ ".png", bbox_inches='tight')
            plt.close()
    sio.savemat(os.path.join(path_save_root,f'{dict_people_type[0]}_{dict_people_type[1]}+_tp_mat.mat'), save_tp_mat)
def get_percentage_mat(p_values,dict_atlas, threshold=0.001):
    n_atlas = len(dict_atlas)
    percentage_mat = np.zeros((n_atlas,n_atlas))
    for i_index, i_key in enumerate(dict_atlas):
        for j_index, j_key in enumerate(dict_atlas):
            p_ij = p_values[np.ix_(dict_atlas[i_key], dict_atlas[j_key])]
            percentage_mat[i_index,j_index] = np.sum(p_ij < threshold) / p_ij.size
    return percentage_mat



def connectivity_altas_ttest_plot_percentage(dict_people_type=['ASD_H', 'TD_P']):
    dict_atlas = get_atlas_net()
    new_sequence = []
    positions = {}
    position = 0

    for i_net in dict_atlas.keys():
        values = dict_atlas[i_net]
        new_sequence.extend(values)
        positions[i_net] = list(range(position, position + len(values)))
        position += len(values)
    ticks = []
    plt.rcParams['font.family'] = 'Times New Roman'
    labels = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
    for name, index_range in positions.items():
        middle = (index_range[0] + index_range[-1]) / 2
        ticks.append(middle)
    fre_band_list = ['jiang_gamma_alpha_max', 'jiang_gamma_theta_max',
                     'jiang_beta_theta_max', 'jiang_beta_alpha_max']
    path_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_new"
    path_save_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_nilearn"
    if not os.path.isdir(path_save_root):
        os.makedirs(path_save_root)
    for i_resting_type in ['ec_avg']:
        for fre_index, fre_band in enumerate(fre_band_list):
            figure_name = dict_people_type[0] + " vs " + dict_people_type[1] + "_"+fre_band
            path_save_fig_t = os.path.join(path_save_root, figure_name)
            p_values = sio.loadmat(os.path.join(path_root, f'{dict_people_type[0]}_{dict_people_type[1]}+_tp_mat.mat'))["p_" + figure_name]
            p_percent_mat = get_percentage_mat(p_values,positions,0.0001)*100
            norm = Normalize(vmin = np.min(p_percent_mat), vmax = np.max(p_percent_mat))
            mapper = cm.ScalarMappable(norm=norm, cmap='Blues')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            # sns.set_context("paper", font_scale=2)
            sns.heatmap(p_values, cmap = "Blues", cbar=True,square=True,ax=ax1,vmin=np.min(p_percent_mat),vmax=np.max(p_percent_mat))
            ax1.set_xticks(ticks);ax1.set_xticklabels(labels)
            ax1.set_yticks(ticks);ax1.set_yticklabels(labels)
            # cmap = ax.collections[0].cmap
            for name, index_range in positions.items():
                start = index_range[0];
                end = index_range[-1]+1
                if name == "Vis":
                    end = end - 1
                ax1.add_patch(plt.Rectangle((start, start), end - start, end - start, fill=True, facecolor='green',
                                            edgecolor='black', lw=0.5))
            for i_index, (i_name, i_index_range) in enumerate(positions.items()):
                for j_index, (j_name, j_index_range) in enumerate(positions.items()):
                    if i_index!=j_index:
                        i_start = i_index_range[0]; j_start=j_index_range[0]
                        i_end = i_index_range[-1] + 1; j_end = j_index_range[-1] + 1
                        if i_index==6 :
                            i_end = i_end-1
                        if  j_index == 6:
                            j_end = j_end-1
                        blue_color = mapper.to_rgba(p_percent_mat[i_index,j_index])
                        ax1.add_patch(plt.Rectangle((i_start, j_start), i_end - i_start, j_end - j_start, fill=True, facecolor=blue_color,edgecolor='black', lw=0.5))
            ax1.set_title("p_" + figure_name)
            ax2 = sns.heatmap(p_percent_mat, cmap="Reds", cbar=True, annot=True,xticklabels=labels,yticklabels=labels,fmt=".2f",square=True)
            plt.savefig(os.path.join(path_save_root,figure_name + "_frequency.png"), bbox_inches='tight')
            plt.close()
def connectivity_altas_ttest_plot_nilearn(coord,used_node_colors,dict_people_type = ['ASD_H', 'TD_P']):
    dict_atlas = get_atlas_net()
    new_sequence = []
    positions = {}
    position = 0
    path_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_new"
    path_save_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_nilearn"
    # colormap_green = cm.get_cmap('Blues_r')
    for i_net in dict_atlas.keys():
        values = dict_atlas[i_net]
        new_sequence.extend(values)
        positions[i_net] = list(range(position, position + len(values)))
        position += len(values)
    ticks = []
    labels = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
    new_dict_atlas = {key: dict_atlas[old_key] for old_key, key in zip(dict_atlas.keys(), labels)}
    sio.savemat(os.path.join(path_save_root,"400_label.mat"),new_dict_atlas)
    for name, index_range in positions.items():
        middle = (index_range[0] + index_range[-1]) / 2
        ticks.append(middle)
    fre_band_list = ['jiang_gamma_alpha_max', 'jiang_gamma_theta_max',
                     'jiang_beta_theta_max', 'jiang_beta_alpha_max']
    dict_resting_type = ['ec_avg']
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    if not os.path.isdir(path_save_root):
        os.makedirs(path_save_root)
    for i_resting_type in dict_resting_type:
        for fre_index, fre_band in enumerate(fre_band_list):
            if fre_band=='jiang_beta_theta_max':
                figure_name =  f'{dict_people_type[0]}__{dict_people_type[1]}_{fre_band}'
                path_save_fig_t = os.path.join(path_save_root,figure_name)
                t_values = sio.loadmat(os.path.join(path_root,f'{dict_people_type[0]}_{dict_people_type[1]}+_tp_mat.mat'))["t_"+figure_name]
                p_values = sio.loadmat(os.path.join(path_root, f'{dict_people_type[0]}_{dict_people_type[1]}+_tp_mat.mat'))["p_" + figure_name]
                t_values[p_values >= 0.0001] = 0
                # print(np.sum(p_values < 0.0001))
                for atlas in new_dict_atlas.keys():
                    t_values[np.ix_(new_dict_atlas[atlas],new_dict_atlas[atlas])]=0
                    p_values[np.ix_(new_dict_atlas[atlas], new_dict_atlas[atlas])]= np.nan

                print(np.sum(p_values < 0.0001))
                fig, (ax1) = plt.subplots(1, 1, figsize=(25, 10))
                edge_kwargs = dict(linewidth = 0.001)
                plotting.plot_connectome(adjacency_matrix = t_values.T, node_coords=coord, node_size=70, edge_cmap=plt.get_cmap('RdBu_r'),
                                         edge_vmin=-6,
                                         edge_vmax=6,#np.nanmax(t_values_positive),
                                         edge_kwargs=edge_kwargs, node_color = used_node_colors, colorbar=True, figure=fig, axes=ax1,title=figure_name)
                # if np.nanmin(t_values_negative) < 0:
                #     plotting.plot_connectome(adjacency_matrix = t_values_negative.T, node_coords=coord, node_size=70, edge_cmap=plt.get_cmap('Blues_r'),
                #                              edge_kwargs=edge_kwargs,
                #                              edge_vmin= -6,#np.nanmin(t_values_negative),
                #                              edge_vmax=0,
                #                              node_color=used_node_colors, colorbar=True, figure=fig, axes=ax2,
                #                              title=figure_name)
                plt.savefig(path_save_fig_t + ".svg", bbox_inches = 'tight')
                plt.close()
def connectivity_altas_ttest_plot_nilearn_LN(coord,used_node_colors,dict_people_type = ['ASD_H', 'TD_P']):
    dict_atlas = get_atlas_net()
    new_sequence = []
    positions = {}
    position = 0
    path_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_new"
    path_save_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot_nilearn"
    for i_net in dict_atlas.keys():
        values = dict_atlas[i_net]
        new_sequence.extend(values)
        positions[i_net] = list(range(position, position + len(values)))
        position += len(values)
    ticks = []
    labels = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
    new_dict_atlas = {key: dict_atlas[old_key] for old_key, key in zip(dict_atlas.keys(), labels)}
    sio.savemat(os.path.join(path_save_root,"400_label.mat"),new_dict_atlas)
    for name, index_range in positions.items():
        middle = (index_range[0] + index_range[-1]) / 2
        ticks.append(middle)
    fre_band_list = ['jiang_gamma_alpha_max', 'jiang_gamma_theta_max',
                     'jiang_beta_theta_max', 'jiang_beta_alpha_max']
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    if not os.path.isdir(path_save_root):
        os.makedirs(path_save_root)
    t_values_all = []; p_values_all = []
    for fre_index, fre_band in enumerate(fre_band_list):
        figure_name =  f'{dict_people_type[0]}__{dict_people_type[1]}_{fre_band}'
        # path_save_fig_t = os.path.join(path_save_root,figure_name)
        t_values = sio.loadmat(os.path.join(path_root,f'{dict_people_type[0]}_{dict_people_type[1]}+_tp_mat.mat'))["t_"+figure_name]
        p_values = sio.loadmat(os.path.join(path_root, f'{dict_people_type[0]}_{dict_people_type[1]}+_tp_mat.mat'))["p_" + figure_name]
        t_values_all.append(t_values); p_values_all.append(p_values)
    t_values_all = np.array(t_values_all); p_values_all = np.array(p_values_all)
    min_p_values = np.min(p_values_all, axis=0)
    min_p_indices = np.argmin(p_values_all, axis=0)
    min_t_values = np.take_along_axis(t_values_all, np.expand_dims(min_p_indices, axis=0), axis=0)[0]
    min_t_values[min_p_values >= 0.0001] = 0
    new_min_t_values = np.zeros_like(min_t_values)
    indices = new_dict_atlas['LN']
    new_min_t_values[indices, :] = min_t_values[indices, :]
    new_min_t_values[:, indices] = min_t_values[:, indices]
    new_min_t_values[np.ix_(indices, indices)]=0
    # for i in range(len(indices)):
    #     new_min_t_values[indices[i], indices[i]] = 0

    if not np.all(new_min_t_values == 0):
        fig, (ax1) = plt.subplots(1, 1, figsize=(25, 10))
        edge_kwargs = dict(linewidth = 0.001)
        plotting.plot_connectome(adjacency_matrix = new_min_t_values.T, node_coords=coord, node_size=70, edge_cmap = plt.get_cmap('RdBu_r'),
                                 edge_vmin=-6,
                                 edge_vmax=6,
                                 edge_kwargs=edge_kwargs, node_color = used_node_colors, colorbar=True, figure=fig, axes=ax1,title= f'{dict_people_type[0]}__{dict_people_type[1]}LN')
        plt.savefig(os.path.join(path_save_root, f'{dict_people_type[0]}_{dict_people_type[1]}_LN_400_400.svg') , bbox_inches = 'tight')
        plt.close()
def plot_connectivity_net_400():
    fs_file_path = r"C:\Users\15956\Desktop\cortex_pial_low.fs"
    vertice_path = r"C:\Users\15956\Desktop\vertice_cell.mat"
    atlas_dict = get_atlas_net()
    colormap = plt.cm.Spectral  # 或者 plt.cm.cool
    roi_colors = to_rgba_array(colormap(np.linspace(0.2, 0.85, len(atlas_dict))))

    positions = get_position_400(fs_file_path,vertice_path)
    positions[:,1] = positions[:,1] - 18
    positions[:, 2] = positions[:, 2] + 20
    node_colors = np.full((400, 4), (0.5, 0.5, 0.5, 1.0), dtype=np.float32)
    used_location = np.zeros((400))
    for i, roi in enumerate(atlas_dict):
        channels = atlas_dict[roi]
        used_location[channels] = 1
        node_colors[channels, :] = roi_colors[i, :]

    used_node_colors = node_colors
    new_t = np.zeros((len(used_node_colors), len(used_node_colors)))
    fig = plt.figure(figsize=(50, 18))
    ax1 = fig.add_axes([0.05, 0.1, 0.7, 0.8])
    plotting.plot_connectome(adjacency_matrix=new_t, node_coords = positions, node_size = 150,
                             node_color=used_node_colors, colorbar = False, figure=fig, axes=ax1)
    roi_names = list(atlas_dict.keys())
    color_blocks = np.array([roi_colors]).reshape(7, 1, 4)
    ax2 = fig.add_axes([0.75, 0.1, 0.01, 0.8])
    ax2.imshow(color_blocks[::-1], aspect='auto', extent = [0, 1, 0, 7])  # 翻转颜色块
    # 添加名称标签
    # for i in range(n_ROI):
    #     ax2.text(1.1, i + 0.5, roi_names[i], ha='left', va='center', fontsize=10)
    ax2.axis('off')
    plt.savefig("dd1.png", bbox_inches="tight")
    plt.close()
    return positions, used_node_colors


if __name__ == "__main__":
    # connectivity_calculate_all()
    for items in [['ASD_P', 'TD_P'], ['ASD_L', 'TD_P'], ['ASD_H', 'TD_P']]:
        connectivity_altas_ttest_plot(items)
    # positions, used_node_colors = plot_connectivity_net_400()
    # for items in [['ASD_P', 'TD_P'], ['ASD_L', 'TD_P'], ['ASD_H', 'TD_P']]:
    #      # connectivity_altas_ttest_plot(items)
    #      # connectivity_altas_ttest_plot_percentage(items)
    #      connectivity_altas_ttest_plot_nilearn(positions,used_node_colors,items)

