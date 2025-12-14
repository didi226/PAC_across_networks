rootPath = 'D:\data\KLC\average\ec_avg';
folderNames = {'ASD_P','ASD_L', 'ASD_H','TD_P','TD_L','TD_H'};
% 创建新的文件夹地址
for i = 1:length(folderNames)
    % 当前文件夹名称
    folderName = folderNames{i};
    % 创建新的文件夹地址
    folderPath = fullfile(rootPath, folderName);
    setFiles = getAllSetFiles(folderPath);
    for i_setFiles = 1:length(setFiles)
        [~, name, ~] = fileparts(setFiles{i_setFiles});
        subjectName = regexprep(name, '\.set$', ''); % 去掉.set后缀
        [rawDataFilePath, rawFileName, ~] = fileparts(setFiles{i_setFiles});
        saveFileName = fullfile(rawDataFilePath, sprintf('scout_400_%s.mat', rawFileName));
        if ~exist(saveFileName, 'file')
            bst_process_data(subjectName, setFiles{i_setFiles});
        end
    end
end

function setFiles = getAllSetFiles(folderPath)
    % 获取指定文件夹下所有.set文件的地址
    % 输入参数：
    % folderPath - 指定文件夹的路径
    % 检查文件夹是否存在
    if ~exist(folderPath, 'dir')
        error('指定的文件夹不存在：%s', folderPath);
    end
    % 获取文件夹中所有.set文件
    files = dir(fullfile(folderPath, '*.set'));
    setFiles = fullfile({files.folder}, {files.name});
    if isempty(setFiles)
        disp('没有找到.set文件');
    end
end
