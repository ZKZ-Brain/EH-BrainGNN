%computing the functional connectivity network (AAL)
for i=1:1642
    load(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/AAL/subj_' num2str(i,'%04d') '_ROI.mat']);
    R=corr(ROI_TC,'type','Pearson');
    R=(R+R')/2;
    R(isnan(R))=0;
    R=R-diag(diag(R));
    R(R>=1)=1-1e-16;
    corrmatrix=(0.5*log((1+R)./(1-R)));
    save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/AAL/subj_' num2str(i,'%04d') '_FC.mat'],'corrmatrix');
    disp(i);
end

%computing the functional connectivity network (Power 264 functional ROIs)
load('/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Pow_patient_idx.mat');
number = length(patient_idx);
for i=1:number
    load(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Power 264 functional ROIs/subj_' num2str(patient_idx(i),'%04d') '_ROI.mat']);
    R=corr(ROI_TC,'type','Pearson');
    R=(R+R')/2;
    R(isnan(R))=0;
    R=R-diag(diag(R));
    R(R>=1)=1-1e-16;
    corrmatrix=(0.5*log((1+R)./(1-R)));
    save(['/Users/xyazkz/Downloads/data/depression_FC(848vs794)/Power 264 functional ROIs/subj_' num2str(patient_idx(i),'%04d') '_FC.mat'],'corrmatrix');
    disp(i);
end
