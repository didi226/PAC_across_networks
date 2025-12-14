from ANT_data_processing.feature_analyse_and_classify.feature_visualization import get_filelist
from sub_pipeline.sub.sub_asd.source_level.source_atlas_asd import get_atlas_net
from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import to_rgba_array
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
def connectivity_altas_net_calculate():
	dict_atlas = get_atlas_net()
	connectivity_altas = np.zeros((4,7,7))
	fre_band_list = ['jiang_gamma_alpha_max','jiang_gamma_theta_max',
	                 'jiang_beta_theta_max','jiang_beta_alpha_max']
	dict_resting_type = ['ec_avg']
	dict_people_type = ['ASD_P', 'ASD_L', 'TD_H', 'ASD_H', 'TD_P', 'TD_L']
	path_root = r"D:\data\KLC\average\sout_atlas_pac_connectivity_mat"
	path_save_root = r"D:\data\KLC\average\net_sout_atlas_pac_connectivity_mean"
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_mat_resting_type = os.path.join(path_root, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		if not os.path.isdir(path_save_resting_type):
			os.makedirs(path_save_resting_type)
		for i_people_type in dict_people_type:
			path_people_type = os.path.join(path_mat_resting_type, i_people_type)
			file_list = get_filelist(path_people_type, p_sort = False)
			path_save_people_type = os.path.join(path_save_resting_type, i_people_type)
			if not os.path.isdir(path_save_people_type):
				os.makedirs(path_save_people_type)
			for i_file in file_list:
				net_connectivity_dic = {}
				print(i_file)
				save_path_file  = os.path.join(path_save_people_type, i_file.rstrip(".mat") +os.path.basename(path_save_root) + ".mat")
				for fre_index, fre_band in enumerate(fre_band_list) :
					mat_data = np.array(sio.loadmat(os.path.join(path_people_type, i_file))[fre_band])
					for i,(atlas_name_i,atlas_temp_i) in enumerate(dict_atlas.items()):
						for j,(atlas_name_j,atlas_temp_j)  in enumerate(dict_atlas.items()):
							connectivity_altas[fre_index,i,j]= np.mean(mat_data[np.ix_(atlas_temp_i,atlas_temp_j)])
				net_connectivity_dic['data'] = connectivity_altas
				net_connectivity_dic['fre_list'] = fre_band_list
				net_connectivity_dic['atlas_list'] =  list(dict_atlas.keys())
				sio.savemat(save_path_file, net_connectivity_dic)
def get_atlas_connectivity_people_type_fre(path_mat_resting_type, i_people_type, fre_band, new_sequence):
	path_people_type = os.path.join(path_mat_resting_type, i_people_type)
	file_list = get_filelist(path_people_type, p_sort = False)
	connectivity_all = np.zeros((len(file_list),400,400))
	for i,i_file in enumerate(file_list):
		mat_data = np.array(sio.loadmat(os.path.join(path_people_type, i_file))[fre_band])
		connectivity_all[i, :, :] = mat_data[new_sequence, :][:, new_sequence]
	return connectivity_all

def connectivity_altas_ttest_plot(dict_people_type = ['ASD_H', 'TD_P']):
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
	labels = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
	for name, index_range in positions.items():
		middle = (index_range[0] + index_range[-1]) / 2
		ticks.append(middle)
	fre_band_list = ['jiang_gamma_alpha_max','jiang_gamma_theta_max',
	                 'jiang_beta_theta_max','jiang_beta_alpha_max']
	dict_resting_type = ['ec_avg']
	path_root = r"D:\data\KLC\average\sout_atlas_pac_connectivity_mat"
	path_save_root = r"C:\Users\15956\Desktop\KLC_results\sout_atlas_pac_connectivity_plot"
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_mat_resting_type = os.path.join(path_root, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		for fre_index, fre_band in enumerate(fre_band_list):
			path_save_fig = os.path.join(path_save_root , dict_people_type[0] + "__vs__" +dict_people_type[1] + fre_band + ".png")
			connectivity_0 = get_atlas_connectivity_people_type_fre(path_mat_resting_type, dict_people_type[0], fre_band,new_sequence)
			connectivity_1 = get_atlas_connectivity_people_type_fre(path_mat_resting_type, dict_people_type[1], fre_band,new_sequence)
			t_values = np.zeros((400,400))
			p_values = np.zeros((400,400))
			for i in range(400):
				for j in range(400):
					t_values[i, j], p_values[ i, j] = stats.ttest_ind(connectivity_0[:,i, j], connectivity_1[:,i, j])
			plt.rcParams['font.family'] = 'Times New Roman'
			plt.figure(figsize=(12, 10))
			sns.set_context("paper", font_scale = 2)
			ax = sns.heatmap(t_values, cmap = "RdBu_r",cbar = True, vmin = -2, vmax = 2)
			ax.set_xticks(ticks)
			ax.set_xticklabels(labels)
			ax.set_yticks(ticks)
			ax.set_yticklabels(labels)
			cmap = ax.collections[0].cmap
			for name, index_range in positions.items():
				start = index_range[0]
				end = index_range[-1]
				ax.add_patch(plt.Rectangle((start, start), end - start, end - start, fill = True, edgecolor = 'white', lw = 3))
				connectivity_net_name_0 = np.mean(connectivity_0[:,index_range,index_range], axis = 1)
				connectivity_net_name_1 = np.mean(connectivity_1[:,index_range,index_range], axis = 1)
				t_value, p_value = stats.ttest_ind(connectivity_net_name_0, connectivity_net_name_1)
				color = cmap((t_value - (-2)) / (2 - (-2)))  # 根据 t_value 在 color map 上的位置选择颜色
				ax.plot([start, end], [start, end], color = color, lw = 3)
			plt.savefig(path_save_fig, bbox_inches = 'tight')




def get_net_pac_atlas(i_people_type,path_mat_resting_type):
	path_people_type = os.path.join(path_mat_resting_type, i_people_type)
	file_list = get_filelist(path_people_type, p_sort=False)
	connectivity_altas = []
	for i_file in file_list:
		mat_data = sio.loadmat(os.path.join(path_people_type, i_file))
		connectivity_altas.append(mat_data['data'])
		fre_band_list = mat_data['fre_list']
		atlas_list = mat_data['atlas_list']
	connectivity_altas= np.array(connectivity_altas)

	cleaned_file_list = []
	for file in file_list:
		# Remove "scout_400"
		cleaned_file = file.replace("scout_400_", "")
		# Remove the part after the required suffix
		cleaned_file = cleaned_file.split("pacnet_sout_atlas_pac_connectivity_mean.mat")[0]
		cleaned_file_list.append(cleaned_file)

	return connectivity_altas,fre_band_list,atlas_list,cleaned_file_list


def create_diagonal_mask(net_7_names, excluded_names):
	"""
	创建一个对角线掩码，将指定名称的行和列置为1，对角线置为0。
	参数:
	- net_7_names: 所有网络名称的列表。
	- excluded_names: 需要剔除的网络名称的列表。
	返回:
	- 对角线掩码数组，形状为 (len(net_7_names), len(net_7_names))。
	"""
	# 创建一个形状为 (len(net_7_names), len(net_7_names)) 的布尔类型数组，初始值都为 False
	diagonal_mask = np.zeros((len(net_7_names), len(net_7_names)), dtype=bool)
	# 获取需要剔除的网络名称对应的索引
	excluded_indices = [net_7_names.index(name) for name in excluded_names]
	# 将对应行列置为1
	diagonal_mask[excluded_indices, :] = True
	diagonal_mask[:, excluded_indices] = True
	# 将对角线上的值置为0
	np.fill_diagonal(diagonal_mask, False)
	return diagonal_mask

def correct_pvalues_excluding_diagonal(p_values, excluded_name = 'LN', correct = True):
	"""
	对排除对角线的p值进行FDR校正，并返回校正后的p值数组。

	参数:
	- p_values: 原始的p值数组，形状为(4, 7, 7)。

	返回:
	- 校正后的p值数组，形状与输入数组相同。
	"""
	#只计算和绘制某些
	# 创建一个掩码，用于选择每个7x7矩阵的非对角线元素
	net_7_names = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
	excluded_names = [excluded_name]
	mask = create_diagonal_mask(net_7_names,excluded_names)
	# mask = np.ones((7, 7), dtype=bool)
	# np.fill_diagonal(mask, False)
	# 应用掩码，提取所有非对角线元素
	p_values_to_correct = p_values[:, mask]
	# 将非对角线元素拉平，进行FDR校正
	if correct:
		_, pvals_corrected_flat, _, _ = multipletests(p_values_to_correct.flatten(), alpha = 0.05, method='fdr_bh')
	else:
		pvals_corrected_flat = p_values_to_correct.flatten()
	# 创建一个全1数组，用于存放校正后的p值（包括对角线元素）
	p_values_corrected = np.ones_like(p_values)
	# 将校正后的非对角线元素放回原数组的相应位置
	p_values_corrected[:, mask] = pvals_corrected_flat.reshape(p_values_to_correct.shape)
	return p_values_corrected




def generate_circos(p_values, t_values,threshold=0.05):
	sectors = { "gamma": 6,"beta": 6,"alpha": 6,"theta": 6 }
	name2color = {"theta": '#264653', "alpha": '#E76F51', "beta": '#2A9D8F', "gamma": '#E9C46A'}
	name2greek = {"theta": r'$\theta$', "alpha": r'$\alpha$', "beta": r'$\beta$', "gamma": r'$\gamma$'}
	net_7_names = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
	fre_band_list = ['gamma_alpha', 'gamma_theta', 'beta_theta', 'beta_alpha']
	circos = Circos(sectors,start = -354,end = 6,space = 12)
	colormap_green = cm.get_cmap('Greens_r')  # 选择一个颜色映射
	colormap_red = cm.get_cmap('Oranges_r')
	normalize = Normalize(vmin=0, vmax=0.05)  # 假设p值范围是0到0.05
	for sector in circos.sectors:
		track = sector.add_track((85, 100))
		track.axis(fc=name2color[sector.name])
		track.text(name2greek[sector.name], color = "white", fontsize = 15)
		pos_list = list(range(0, int(track.size + 1)))
		labels = net_7_names
		track.xticks(
			pos_list,
			labels,
			outer=False,
			label_size=15,
			tick_length=1,
			label_margin=3,
			label_orientation="vertical",
		)
		line_track = sector.add_track((65, 85), r_pad_ratio=0.1)
	for k in range(4):
		for i in range(7):
			for j in range(7):
				if i==j:
					continue
				if p_values[k, i, j] < threshold:
					bands = fre_band_list[k].split('_')
					if bands[1] == "theta":
						circos.link_line((bands[1], j), (bands[0], i), color = colormap_green(normalize(p_values[k, i, j])),direction=1,arrow_width=5,lw=4)
					if bands[1] == "alpha":
						circos.link_line((bands[1], j), (bands[0], i), color= colormap_red(normalize(p_values[k, i, j])),direction=1,arrow_width=5,lw=4)

	fig = circos.plotfig()
	return fig
def generate_circos_2(p_values, t_values,threshold=0.05):
	dict_atlas = get_atlas_net()
	positive_values = t_values[t_values > 0]
	net_7_names = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
	sectors = {name: len(dict_atlas[key]) for name, key in zip(net_7_names, dict_atlas)}
	colormap = plt.cm.Spectral
	roi_colors = to_rgba_array(colormap(np.linspace(0.2, 0.85, len(net_7_names))))
	name2color = dict(zip(net_7_names, roi_colors))
	name2greek = {"theta": r'$\theta$', "alpha": r'$\alpha$', "beta": r'$\beta$', "gamma": r'$\gamma$'}
	name2 = list(name2greek.keys())


	fre_band_list = ['gamma_alpha', 'gamma_theta', 'beta_theta', 'beta_alpha']
	circos = Circos(sectors,start = -356,end = 4,space = 8)
	colormap_green = cm.get_cmap('Greens_r')  # 选择一个颜色映射
	# colormap_red = cm.get_cmap('Oranges_r')
	normalize = Normalize(vmin=np.min(positive_values), vmax=np.max(positive_values))  # 假设p值范围是0到0.05
	for index, sector in enumerate(circos.sectors):
		track = sector.add_track((97, 100))
		track.axis(fc = name2color[sector.name])
		track.text(sector.name, r=105, color="black", fontsize=20)
		# track.text(net_7_names[index], color = "black", fontsize = 15)
		pos_list = np.linspace(0, list(sectors.values())[index], 4)
		labels = list(name2greek.values())
		track.xticks(
			pos_list,
			labels,
			outer=False,
			label_size=20,
			tick_length=1,
			label_margin=2,
			label_orientation="vertical",
		)
		line_track = sector.add_track((87, 97), r_pad_ratio=0)
	for k in range(4):
		for i in range(7):
			for j in range(7):
				if i==j:
					continue
				if p_values[k, i, j] < threshold:
					bands = fre_band_list[k].split('_')
					if bands[1] == "theta" or bands[1] == "alpha":
						if t_values[k, i, j] >= 0:
							circos.link_line((net_7_names[j], name2.index(bands[1]) * sectors[net_7_names[j]] / 3),
											 (net_7_names[i], name2.index(bands[0]) * sectors[net_7_names[i]] / 3),
											 color = '#D1C6E7', direction=1,
											 arrow_width=5, lw= normalize(abs(t_values[k,i, j]) * 10))
						else:

							circos.link_line((net_7_names[j], name2.index(bands[1])*sectors[net_7_names[j]]/3),
											 (net_7_names[i],name2.index(bands[0])*sectors[net_7_names[i]]/3),
											 color = '#FFC0CB',direction=1,arrow_width=5,lw=-0.5 + normalize(abs(t_values[k,i, j]) * 10))
					# if bands[1] == "alpha":
					# 	circos.link_line((net_7_names[j],name2.index(bands[1])*sectors[net_7_names[j]]/3),
					# 					 (net_7_names[i],name2.index(bands[0])*sectors[net_7_names[i]]/3),
					# 					 color= colormap_green(normalize(p_values[k, i, j])),direction=1,arrow_width=5,lw=4)

	fig = circos.plotfig()
	return fig

def connectivity_altas_net_ttest(dict_people_type, include_name = 'LN',corre = True, circos = True,threshold=0.05):
	dict_resting_type = ['ec_avg']
	path_root = r"D:\data\KLC\average\net_sout_atlas_pac_connectivity_mean"
	for i_resting_type in dict_resting_type:
		path_mat_resting_type = os.path.join(path_root, i_resting_type)
		connectivity_altas0, fre_band_list, atlas_list,_ = get_net_pac_atlas(dict_people_type[0],path_mat_resting_type)
		connectivity_altas1, fre_band_list, atlas_list,_ = get_net_pac_atlas(dict_people_type[1], path_mat_resting_type)
		atlas_list = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
		p_values = np.zeros((4,7,7))
		t_values = np.zeros((4, 7, 7))
		for k in range(4):
			for i in range(7):
				for j in range(7):
					t_values[k, i, j], p_values[k, i, j] = stats.ttest_ind(connectivity_altas0[:,k, i, j], connectivity_altas1[:, k, i, j])
		path_save_mat = os.path.join(r'C:\Users\15956\Desktop\KLC_results\ttest_pac_net_mean_all', #include_name +
									 i_resting_type + "_" + dict_people_type[0] +"__" + dict_people_type[
										 1] + '.mat')  # non_cor
		p_values = correct_pvalues_excluding_diagonal(p_values,include_name,corre)
		savemat(path_save_mat,{'p_values': p_values, 't_values': t_values, 'atlas_list': atlas_list, 'fre_band_list': fre_band_list})

		if circos:
			fig = generate_circos_2(p_values, t_values,threshold=threshold)
			path_save_fig = os.path.join(r'C:\Users\15956\Desktop\KLC_results\ttest_pac_net_mean_all', #include_name +
			                             str(threshold*100)+i_resting_type + "_" + dict_people_type[0] + " vs " + dict_people_type[1]+'.png')#non_cor cor
			plt.savefig(path_save_fig, bbox_inches = 'tight')

		for i, condition in enumerate(fre_band_list):
			plt.rcParams['font.family'] = 'Times New Roman'
			plt.figure(figsize = (8, 6))  # 设置图形大小
			# mask = p_values[i] >= 0.05
			sns.heatmap(t_values[i], xticklabels = atlas_list, yticklabels = atlas_list, vmin = -4, vmax = 4,cmap = "RdBu_r")
			plt.title(f'T-Values Heatmap for {condition}')
			path_save_fig = os.path.join(r'C:\Users\15956\Desktop\KLC_results\ttest_pac_net_mean_t_values',
			                              dict_people_type[0] + " vs " + dict_people_type[1] +
			                             i_resting_type + '_' + condition + 't_value.png')
			plt.savefig(path_save_fig, bbox_inches = 'tight')
			plt.close()
def save_all_net_data():
	dict_resting_type = ['ec_avg']
	path_root = r"D:\data\KLC\average\net_sout_atlas_pac_connectivity_mean"
	connectivity_altas = []
	people_type_list=[]
	file_list_all=[]
	for i_resting_type in dict_resting_type:
		for i_people_type in ['ASD_P','ASD_L','ASD_H','TD_P','TD_L','TD_H']:
			path_mat_resting_type = os.path.join(path_root, i_resting_type)
			connectivity_altas0, fre_band_list, atlas_list, file_list = get_net_pac_atlas(i_people_type,path_mat_resting_type)
			people_type_list.extend([i_people_type]*len(file_list))
			connectivity_altas.extend(connectivity_altas0)
			file_list_all.extend(file_list)
	data_dict = {
		'connectivity_altas': connectivity_altas,
		'file_list_all': file_list_all,
		'people_type_list': people_type_list,
		'fre_band_list': fre_band_list,
		'atlas_list': atlas_list
	}

	# Save data
	savemat('net_data.mat', data_dict)




if __name__ == "__main__":
	dict_people_type = ['ASD_H', 'TD_P']
	connectivity_altas_net_ttest(dict_people_type, include_name='LN', corre = False, circos = True,threshold=0.01)
	dict_people_type = ['ASD_P', 'TD_P']
	connectivity_altas_net_ttest(dict_people_type, include_name='LN', corre = False, circos = True,threshold=0.01)
	dict_people_type = ['ASD_L', 'TD_P']
	connectivity_altas_net_ttest(dict_people_type, include_name='LN', corre = False, circos = True,threshold=0.01)
	# for i in ['ASD_P','ASD_L','ASD_H','TD_H']:
	# 	dict_people_type[0] = i
	# 	connectivity_altas_net_ttest(dict_people_type)
	# connectivity_altas_net_calculate()
	# connectivity_altas_ttest_plot(dict_people_type = ['ASD_P', 'TD_P'])
	# save_all_net_data()