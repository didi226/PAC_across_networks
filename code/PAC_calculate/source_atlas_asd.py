import scipy.io as sio
import os
import re
import numpy as np
import scipy
from scuteegfe.mne_features_wrapper.feature_wrapper import Feature
from ANT_data_processing.feature_analyse_and_classify.feature_visualization import get_filelist
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sub_pipeline.sub.sub_asd.source_level.memd.MEMD_all import memd
from einops import rearrange
from sub_pipeline.sub.sub_asd.source_level.s_estimator import SEstimator
from mne.filter import filter_data
from statsmodels.stats.multitest import multipletests


def calculate_pac_scout_atlas():
	dict_fre_band = {"gamma_alpha": np.array([[8, 12], [30, 45]]), "gamma_theta": np.array([[4, 8], [30, 45]]),
	                 "beta_theta": np.array([[4, 8], [12, 30]]), "beta_alpha": np.array([[8, 12], [12, 30]])}
	dict_resting_type = ['ec_avg', 'eo_avg']
	dict_people_type = ['ASD_P', 'ASD_L', 'ASD_H', 'TD_P', 'TD_L', 'TD_H']
	path_scout_original_mat = r"D:\data\KLC\average\sout_atlas_original_mat"
	path_save_root = r"D:\data\KLC\average\sout_atlas_pac_mat"
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_scout_original_mat_resting_type = os.path.join(path_scout_original_mat, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		if not os.path.isdir(path_save_resting_type):
			os.makedirs(path_save_resting_type)
		for i_people_type in dict_people_type:
			path_people_type = os.path.join(path_scout_original_mat_resting_type, i_people_type)
			path_save_people_type = os.path.join(path_save_resting_type, i_people_type)
			if not os.path.isdir(path_save_people_type):
				os.makedirs(path_save_people_type)
			file_list = get_filelist(path_people_type, p_sort = False)
			for i_file in file_list:
				save_path_file = os.path.join(path_save_people_type, i_file.rstrip(".mat") + "pac" + ".mat")
				if os.path.exists(save_path_file):
					continue
				mat_data = scipy.io.loadmat(os.path.join(path_people_type, i_file))['ScoutData'][None, :, :]
				print(mat_data.shape)
				dict_save = {}
				for method in ['tort', 'jiang']:
					for approach_pac in ['max']:
						for key in dict_fre_band:
							fea1 = Feature(mat_data, sfreq=250, selected_funcs=['pac_connectivity'],
							               funcs_params={'pac_connectivity__method': method,
							                             'pac_connectivity__band': dict_fre_band[key],
							                             'pac_connectivity__approach_pac': approach_pac}, n_jobs=-1)
							con_mat = np.squeeze(np.mean(fea1.features, axis=0))
							print(con_mat.shape)
							dict_save[method + "_" + key + "_" + approach_pac] = con_mat

				sio.savemat(save_path_file, dict_save)

def calculate_pac_scout_atlas_connectivity():
	dict_fre_band = {"gamma_alpha": np.array([[8, 12], [30, 45]]), "gamma_theta": np.array([[4, 8], [30, 45]]),
	                 "beta_theta": np.array([[4, 8], [12, 30]]), "beta_alpha": np.array([[8, 12], [12, 30]])}
	dict_resting_type = ['ec_avg', 'eo_avg']
	dict_people_type = ['TD_L','ASD_P', 'ASD_L', 'ASD_H', 'TD_P', 'TD_H']
	path_scout_original_mat = r"D:\data\KLC\average\sout_atlas_original_mat"
	path_save_root = r"D:\data\KLC\average\sout_atlas_pac_connectivity_mat"
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_scout_original_mat_resting_type = os.path.join(path_scout_original_mat, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		if not os.path.isdir(path_save_resting_type):
			os.makedirs(path_save_resting_type)
		for i_people_type in dict_people_type:
			path_people_type = os.path.join(path_scout_original_mat_resting_type, i_people_type)
			path_save_people_type = os.path.join(path_save_resting_type, i_people_type)
			if not os.path.isdir(path_save_people_type):
				os.makedirs(path_save_people_type)
			file_list = get_filelist(path_people_type, p_sort=False)
			file_list.reverse()
			for i_file in file_list:
				save_path_file = os.path.join(path_save_people_type, i_file.rstrip(".mat") + "pac" + ".mat")
				if os.path.exists(save_path_file):
					continue
				mat_data = scipy.io.loadmat(os.path.join(path_people_type, i_file))['ScoutData'][None, :, :]
				print(mat_data.shape)
				dict_save = {}
				for method in ['jiang']:
					for approach_pac in ['max']:
						for key in dict_fre_band:
							fea1 = Feature(mat_data, sfreq = 250, selected_funcs=['pac_connectivity'],
							               funcs_params={'pac_connectivity__method': method,
							                             'pac_connectivity__band': dict_fre_band[key],
							                             'pac_connectivity__mode':"non-self",
							                             'pac_connectivity__approach_pac': approach_pac}, n_jobs=-1)
							con_mat = np.squeeze(np.mean(fea1.features, axis=0))
							print(con_mat.shape)
							dict_save[method + "_" + key + "_" + approach_pac] = con_mat

				sio.savemat(save_path_file, dict_save)



def get_atlas_net():
	path = r"C:\Users\15956\Desktop\pipline\net7_400_atlas.mat"
	mat_data = np.squeeze(sio.loadmat(path)['labels'])
	extracted_list = [item[0].split('_')[0] for item in mat_data]
	element_positions = {}
	for index, element in enumerate(extracted_list):
		if element in element_positions:
			element_positions[element].append(index)
		else:
			element_positions[element] = [index]
	unique_element_count = len(element_positions)
	return element_positions


def get_atlas_pac(file_name):
	"""
	:param file_name:  str  atlas_pac.mat 文件的地址
	 matrix 8*400 400对应400个区域
	 variable_names 对应的频带和方法名称
	:return:
	"""
	mat = scipy.io.loadmat(file_name)
	variable_names = ['jiang_beta_alpha_max', 'jiang_beta_theta_max', 'jiang_gamma_alpha_max', 'jiang_gamma_theta_max',
	                  'tort_beta_alpha_max', 'tort_beta_theta_max', 'tort_gamma_alpha_max', 'tort_gamma_theta_max']
	variables = []
	for name in variable_names:
		variable = mat[name]
		variables.append(variable)
	matrix = np.squeeze(np.array(variables))
	return matrix, variable_names


def get_atlas_pac_7net(net_positions_dict, pac_mat):
	pac_net_feature = np.zeros((8, 7))
	i = 0
	net_names = []
	for net_name, net_positions in net_positions_dict.items():
		pac_net_feature[:, i] = np.mean(pac_mat[:, net_positions], axis=1)
		i = i + 1
		net_names.append(net_name)
	return pac_net_feature, net_names


def get_atlas_pac_7net_all():
	"""
	:return:
	dict_save 所有的7net的网络都存在dict_save
	"""
	net_positions_dict = get_atlas_net()
	path_root = r"I:\data\KLC\average\sout_atlas_pac_mat"
	dict_resting_type = ['ec_avg', 'eo_avg']
	dict_people_type = ['ASD_P', 'ASD_L', 'ASD_H', 'TD_P', 'TD_L', 'TD_H']
	dict_save = {}
	for i_resting_type in dict_resting_type:
		path_root_resting_type = os.path.join(path_root, i_resting_type)
		for i_people_type in dict_people_type:
			print(i_people_type)
			path_root_people_type = os.path.join(path_root_resting_type, i_people_type)
			file_list = get_filelist(path_root_people_type, p_sort=False)
			print(file_list)
			feature_pac_people_type = np.zeros((len(file_list), 8, 7))
			for file_index, i_file in enumerate(file_list):
				pac_matrix, pac_names = get_atlas_pac(os.path.join(path_root_people_type, i_file))
				feature_pac_people_type[file_index, :, :], net_names = get_atlas_pac_7net(net_positions_dict,
				                                                                          pac_matrix)
			dict_save[i_resting_type + "_" + i_people_type + "pac_net"] = feature_pac_people_type
			dict_save[i_resting_type + "_" + i_people_type + "file_list"]= file_list
	dict_save['pac_names'] = pac_names
	dict_save['net_names'] = net_names
	return dict_save


def load_features(dict_save, state, people_type, suffix):
	"""
	:param dict_save:      dict_save 所有类型的人的PAC的7net所有值
	:param state:          0 close  1 open
	:param people_type:    人的种类
	:param suffix:         "pac_net"
	:return:               某个状态某种人的pac 7net的数据
	"""
	return dict_save[f"{state}_{people_type}{suffix}"]


def create_heatmap(df, title, save_path):
	"""
	绘制热力图并且保存
	:param df:
	:param title:
	:param save_path:
	:return:
	"""
	plt.figure(figsize=(25, 5))
	mask = df >= 0.05
	sns.heatmap(df, annot=True, mask=mask, cmap='viridis', vmin=0, vmax=0.05)
	plt.title(title)
	plt.savefig(save_path, bbox_inches='tight')

def dataframe_corrected(df):
	pvals_first_half = df.iloc[:4, :].values.flatten()
	pvals_second_half = df.iloc[4:, :].values.flatten()

	# 对这两个数组分别进行p值校正
	_, pvals_first_half, _, _ = multipletests(pvals_first_half, method='fdr_bh')
	_, pvals_second_half, _, _ = multipletests(pvals_second_half, method='fdr_bh')

	# 将校正后的p值放回原DataFrame
	df.iloc[:4, :] = pvals_first_half.reshape(4, -1)
	df.iloc[4:, :] = pvals_second_half.reshape(4, -1)
	return df
# Simplified Function
def ttest_pac_all(dict_save, ttest_people_type):
	"""
	:param dict_save:            dict_save 所有类型的人的PAC的7net所有值
	:param ttest_people_type:    list 人的种类  例如['ASD_H', 'TD_P']
	:return:                     两类人不同的网络PAC　ｔｔｅｓｔ的结果　图和表格都保存了
	"""
	dict_resting_type = ['ec_avg', 'eo_avg']
	suffix = "pac_net"
	pac_names = dict_save['pac_names']
	net_names_modify = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
	p_values = np.zeros((16, 7)); t_values = np.zeros((16, 7))
	for i in range(16):
		state = dict_resting_type[i // 8]
		feature1 = load_features(dict_save, state, ttest_people_type[0], suffix)
		feature2 = load_features(dict_save, state, ttest_people_type[1], suffix)
		for j in range(7):
			t_values[i, j], p_values[i, j] = stats.ttest_ind(feature1[:, i % 8, j], feature2[:, i % 8, j])

	pac_names_resting_type = [var + '_close' for var in pac_names] + [var + '_open' for var in pac_names]
	p_values_df = pd.DataFrame(p_values, index = pac_names_resting_type, columns = net_names_modify)
	t_values_df = pd.DataFrame(t_values, index = pac_names_resting_type, columns = net_names_modify)
	filtered_df = p_values_df[p_values_df.index.str.contains('jiang')]
	filtered_df = dataframe_corrected(filtered_df)
	title_name = f"{ttest_people_type[0]} vs {ttest_people_type[1]}"
	save_folder = r'C:\Users\15956\Desktop\KLC_results\source_pac_ttest_abs'
	create_heatmap(filtered_df, title_name, os.path.join(save_folder, title_name + ".png"))
	filtered_df.to_excel(os.path.join(save_folder, title_name + "p.xlsx"), index=True)
	t_values_df.to_excel(os.path.join(save_folder, title_name + "t.xlsx"), index=True)
def emed_source_signal():
	dict_resting_type = ['ec_avg', 'eo_avg']
	dict_people_type = ['ASD_P', 'ASD_L', 'TD_H','ASD_H', 'TD_P', 'TD_L']
	path_scout_original_mat = r"D:\data\KLC\average\sout_atlas_original_mat"
	path_save_root = r"D:\data\KLC\average\sout_atlas_memd"
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_scout_original_mat_resting_type = os.path.join(path_scout_original_mat, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		if not os.path.isdir(path_save_resting_type):
			os.makedirs(path_save_resting_type)
		for i_people_type in dict_people_type:
			path_people_type = os.path.join(path_scout_original_mat_resting_type, i_people_type)
			path_save_people_type = os.path.join(path_save_resting_type, i_people_type)
			if not os.path.isdir(path_save_people_type):
				os.makedirs(path_save_people_type)
			file_list = get_filelist(path_people_type, p_sort = False)
			for i_file in file_list:
				save_path_file = os.path.join(path_save_people_type, i_file.rstrip(".mat") + "memd" + ".mat")
				if os.path.exists(save_path_file):
					continue
				mat_data = np.array(scipy.io.loadmat(os.path.join(path_people_type, i_file))['ScoutData'])*10**10
				print(mat_data.shape)
				#计算memd
				imf_data = memd(mat_data)*10**(-10)
				print(imf_data.shape)
				sio.savemat(save_path_file,  {'imf_data': imf_data})

def filter_source_signal():
	dict_resting_type = ['ec_avg', 'eo_avg']
	dict_people_type = ['ASD_P', 'ASD_L', 'TD_H','ASD_H', 'TD_P', 'TD_L']
	path_scout_original_mat = r"D:\data\KLC\average\sout_atlas_original_mat"
	path_save_root = r"D:\data\KLC\average\sout_atlas_filter"
	dict_fre_band = {"gamma": np.array([30, 45]),  "beta": np.array([12, 30]), "alpha": np.array([8, 12]),
	                 "theta": np.array([4, 8]),"delta": np.array([0.5, 4])}
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_scout_original_mat_resting_type = os.path.join(path_scout_original_mat, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		if not os.path.isdir(path_save_resting_type):
			os.makedirs(path_save_resting_type)
		for i_people_type in dict_people_type:
			path_people_type = os.path.join(path_scout_original_mat_resting_type, i_people_type)
			path_save_people_type = os.path.join(path_save_resting_type, i_people_type)
			if not os.path.isdir(path_save_people_type):
				os.makedirs(path_save_people_type)
			file_list = get_filelist(path_people_type, p_sort = False)
			sfreq = 250
			for i_file in file_list:
				save_path_file = os.path.join(path_save_people_type, i_file.rstrip(".mat") + "memd" + ".mat")
				if os.path.exists(save_path_file):
					continue
				mat_data = np.array(scipy.io.loadmat(os.path.join(path_people_type, i_file))['ScoutData'])*10**10
				print(mat_data.shape)
				imf_data=np.zeros((5,mat_data.shape[0],mat_data.shape[1]))
				for index,bands in enumerate(dict_fre_band.values()):
						imf_data[index,:,:] = filter_data(mat_data, sfreq, bands[0], bands[1])
				sio.savemat(save_path_file,  {'imf_data': imf_data})

def get_net_s_estimators():
	dict_resting_type = ['ec_avg', 'eo_avg']
	dict_people_type = ['ASD_P', 'ASD_L', 'TD_H', 'ASD_H', 'TD_P', 'TD_L']
	#path_root = r"D:\data\KLC\average\sout_atlas_memd"
	path_root = r"D:\data\KLC\average\sout_atlas_filter"
	dict_7net = get_atlas_net()
	# path_save_root = r"D:\data\KLC\average\memd_7net"
	path_save_root = r"D:\data\KLC\average\filter_7net"
	if not os.path.isdir(path_save_root):
		os.makedirs(path_save_root)
	for i_resting_type in dict_resting_type:
		path_memd_resting_type = os.path.join(path_root, i_resting_type)
		path_save_resting_type = os.path.join(path_save_root, i_resting_type)
		if not os.path.isdir(path_save_resting_type):
			os.makedirs(path_save_resting_type)
		for i_people_type in dict_people_type:
			path_people_type = os.path.join(path_memd_resting_type, i_people_type)
			file_list = get_filelist(path_people_type, p_sort = False)
			path_save_people_type = os.path.join(path_save_resting_type, i_people_type)
			if not os.path.isdir(path_save_people_type):
				os.makedirs(path_save_people_type)
			for i_file in file_list:
				print(i_file)
				save_path_file = os.path.join(path_save_people_type, i_file.rstrip(".mat") +os.path.basename(path_save_root) + ".mat")
				# if os.path.exists(save_path_file):
				# 	continue
				imf_data = np.array(scipy.io.loadmat(os.path.join(path_people_type, i_file))['imf_data'])
				imf_data = imf_data[:5,:,:]
				imf_data_dic={}
				for i, (key, value) in enumerate(dict_7net.items()):
					imf_data_dic[key] = imf_data[:,value,:]
				svalues = np.zeros((5,5,7,7))
				all_key=[]
				for i, (i_key, i_value) in enumerate(imf_data_dic.items()):
					for j, (j_key, j_value) in enumerate(imf_data_dic.items()):
						for i_band in range(5):
							for j_band in range(5):
								sestimators = SEstimator(i_value[i_band, :, :], j_value[j_band, :, :])
								svalues[i_band,j_band,i,j] = sestimators.fit()
				# 	all_key.append(i_key)
				# print(all_key)
				imf_data_dic['svalues']= svalues
				sio.savemat(save_path_file, imf_data_dic)
def all_svalues_get(path_memd_resting_type,i_people_type):
	"""
	:param path_memd_resting_type:   str  睁闭眼的地址
	:param i_people_type:            str  人群
	:return:
	"""
	path_people_type = os.path.join(path_memd_resting_type, i_people_type)
	file_list = get_filelist(path_people_type, p_sort=False)
	svalues_all = np.zeros((len(file_list), 5,5, 7, 7))
	for i_index, i_file in enumerate(file_list):
		svalues_all[i_index, :, :, :, :] = np.array(scipy.io.loadmat(os.path.join(path_people_type, i_file))['svalues'])
	return svalues_all

def memd_7net_ttest(dict_people_type):
	"""
    提取memd_7net的关系估计  在不同的人群中做ttest
	:return:
	"""
	# dict_people_type = ['TD_H', 'TD_P']
	dict_resting_type = ['ec_avg', 'eo_avg']
	#path_root = r"D:\data\KLC\average\memd_7net"
	path_root = r"D:\data\KLC\average\filter_7net"
	for i_resting_type in dict_resting_type:
		path_memd_resting_type = os.path.join(path_root, i_resting_type)
		svalues_all_0 = all_svalues_get(path_memd_resting_type,dict_people_type[0])
		svalues_all_1 = all_svalues_get(path_memd_resting_type, dict_people_type[1])
		print(svalues_all_0.shape)
		svalues_all_0 = rearrange(svalues_all_0,'a b c d e -> a (d b) (e c)',a=svalues_all_0.shape[0],b=5,c=5,d=7,e=7)    #(b d) (c e)
		svalues_all_1 = rearrange(svalues_all_1, 'a b c d e -> a (d b) (e c)', a=svalues_all_1.shape[0], b=5, c=5, d=7,e=7) #(b d)
		print(svalues_all_0.shape)
		ttest_matrix = np.zeros((35,35))
		for i in range(35):
			for j in range(35):
				_, ttest_matrix[i,j]=stats.ttest_ind(svalues_all_0[:,i,j],svalues_all_1[:,i,j])
		mask = ttest_matrix >= 0.05
		band_name =['gamma','beta','alpha','theta','delta']
		net_name =['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
		#combined_list = [f"{band}_{net}" for band in band_name for net in net_name]
		combined_list = [f"{band}_{net}"  for net in net_name for band in band_name]
		title_name = dict_people_type[0] + "  vs " + dict_people_type[1] + i_resting_type
		sns.heatmap(ttest_matrix,mask=mask,xticklabels=combined_list,yticklabels=combined_list,vmin=0,vmax=0.05)
		plt.title(title_name)
		#path_save_fig = os.path.join(r'C:\Users\15956\Desktop\KLC_results\ttest_memd',title_name +'.png')
		path_save_fig = os.path.join(r'C:\Users\15956\Desktop\KLC_results\ttest_filter',title_name +'.png')
		plt.savefig(path_save_fig, bbox_inches='tight')
		plt.close()

def extract_feature(file_name):
    match = re.search(r'^[^_]*_[^_]*_([^_]+)_', file_name)
    if match:
        return match.group(1)
    return file_name
def get_DMN_LN_VN_beta_theta(dict_save,people_type=['ASD_P','ASD_L','ASD_H','TD_P','TD_L','TD_H']):
		dict_resting_type = ['ec_avg']
		suffix = "pac_net"
		pac_names = dict_save['pac_names']
		net_names_modify = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
		dict_feature = {}
		for state in dict_resting_type:
			dict_file_name = []
			for i_people_type in people_type:
				feature = load_features(dict_save, state, i_people_type, suffix)[:,1,[1,3,6]]
				file_list =  load_features(dict_save, state, i_people_type, 'file_list')
				file_list = [extract_feature(file) for file in file_list]
				dict_feature[(state, i_people_type)] = feature
				df = pd.DataFrame({i_people_type: file_list})
				dict_file_name.append(df)
			result_df = pd.concat(dict_file_name, axis=1)
			result_df.to_excel(r'C:\Users\15956\Desktop\KLC_results\file_name.xlsx', index=False)

def get_LN_(dict_save, people_type=['ASD_P', 'ASD_L', 'ASD_H', 'TD_P','TD_L','TD_H']):
	dict_resting_type = ['ec_avg']
	suffix = "pac_net"
	pac_names = dict_save['pac_names']
	net_names_modify = ['FPN', 'DMN', 'DAN', 'LN', 'VAN', 'SMN', 'VN']
	fre_band_names =['jiang_beta_alpha_max',
	 'jiang_beta_theta_max',
	 'jiang_gamma_alpha_max',
	 'jiang_gamma_theta_max']
	 # 'tort_beta_alpha_max',
	 # 'tort_beta_theta_max',
	 # 'tort_gamma_alpha_max',
	 # 'tort_gamma_theta_max']

	dict_feature = {}
	file_list = []
	for state in dict_resting_type:
		for i_people_type in people_type:
			file_list.extend(dict_save[f'{state}_{i_people_type}file_list'])
			feature = load_features(dict_save, state, i_people_type, suffix)#[:, [0,2,3],  0]
			dict_feature[(state, i_people_type)] = feature
	processed_file_list = [filename.replace('scout_400_', '').replace('pac.mat', '') for filename in file_list]
	df = pd.DataFrame()
	for people_type, feature in dict_feature.items():
		n_people, n_fre_band, n_net = feature.shape
		people_type_column = [people_type[1]] * n_people
		row_data = {'people_type': people_type_column}
		for k in range(n_net):
			for j in range(len(fre_band_names)):
				col_name = f'{net_names_modify[k]}_{fre_band_names[j]}'
				row_data[col_name] = feature[:, j, k]
		df = pd.concat([df, pd.DataFrame(row_data)], ignore_index=True)
	df['filename'] = processed_file_list
	df = df[['filename'] + [col for col in df.columns if col != 'filename']]
	output_file = 'intra_data.xlsx'
	df.to_excel(output_file, index=False, engine='openpyxl')









if __name__ == "__main__":
	# filter_source_signal()
	# get_net_s_estimators()
	# calculate_pac_scout_atlas()
	dict_save = get_atlas_pac_7net_all()
	# get_DMN_LN_VN_beta_theta(dict_save)
	get_LN_(dict_save)
	# ttest_pac_all(dict_save, ['ASD_H', 'TD_P'])
	# ttest_pac_all(dict_save, ['ASD_L', 'TD_P'])
	ttest_pac_all(dict_save, ['ASD_P', 'TD_P'])

	# emed_source_signal()
	# get_net_s_estimators()
	# memd_7net_ttest( ['ASD_H', 'TD_P'])
	# memd_7net_ttest(['ASD_L', 'TD_P'])
	# memd_7net_ttest(['ASD_P', 'TD_P'])
	# calculate_pac_scout_atlas_connectivity()
