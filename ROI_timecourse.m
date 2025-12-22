namelist  = dir('/Users/xyazkz/Downloads/data/ROISignals_FunImgARglobalCWF/*.mat');
load('/Users/xyazkz/Downloads/PaperScripts-master/Yan_2019_PNAS/StatsSubInfo/Stat_Sub_Info_848MDDvs794NC.mat');

name=cell(2428,1);
for i=1:length(namelist)
    name{i}=namelist(i).name(12:end-4);
end
patient_idx =[];
m=1;
for i=1:1642
    Index = find(ismember(name, SubID{i}));
    load(['ROISignals_FunImgARglobalCWF/' namelist(Index).name]);
%     ROI_TC=ROISignals(:,1:116);
%     save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/AAL/subj_' num2str(i,'%04d') '_ROI.mat'],'ROI_TC');
%     ROI_TC=ROISignals(:,117:228);
%     save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Harvard-Oxford atlas/subj_' num2str(i,'%04d') '_ROI.mat'],'ROI_TC');
%     ROI_TC=ROISignals(:,229:428);
%     save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Craddock clustering 200 ROIs/subj_' num2str(i,'%04d') '_ROI.mat'],'ROI_TC');
%     ROI_TC=ROISignals(:,429:1408);
%     save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Zalesky random parcelations/subj_' num2str(i,'%04d') '_ROI.mat'],'ROI_TC');
%     ROI_TC=ROISignals(:,1409:1568);
%     save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Dosenbach 160 functional ROIs/subj_' num2str(i,'%04d') '_ROI.mat'],'ROI_TC');
    if length(ROISignals(1,:))>1570
        ROI_TC=ROISignals(:,1570:1833);
        save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Power 264 functional ROIs/subj_' num2str(i,'%04d') '_ROI.mat'],'ROI_TC');
        patient_idx(m) = i;
        m= m+1;
    end
    disp(i);
end
patient_idx = patient_idx';
save('/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Pow_patient_idx.mat','patient_idx');