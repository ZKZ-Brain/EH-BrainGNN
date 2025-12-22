function plot_edge_effectsize_heatmaps(FC, FC_combat, label, outPrefix, roiNames)
% Plot edge-wise Cohen's d heatmaps for MDD vs HC (before/after ComBat).
%
% Inputs
%   FC         : nROI x nROI x nSub  (raw FC)
%   FC_combat  : nROI x nROI x nSub  (harmonized FC)
%   label      : nSub x 1            (1=MDD, 0=HC; or categorical/strings)
%   outPrefix  : string              (e.g., 'FigS_ComBat_dHeatmap')
%   roiNames   : (optional) cellstr  (nROI x 1 ROI names)
%
% Outputs
%   Saves PDF and 600-dpi TIFF.

arguments
    FC (:,:,:) double
    FC_combat (:,:,:) double
    label
    outPrefix (1,1) string = "d_heatmaps"
    roiNames = []
end

% ---------- Basic checks ----------
[nROI1,nROI2,nSub] = size(FC);
assert(nROI1==nROI2, 'FC must be square in first two dims.');
assert(all(size(FC_combat)==size(FC)), 'FC_combat must match FC size.');
assert(numel(label)==nSub, 'label length must match number of subjects.');

% ---------- Normalize labels to logical indices ----------
[labelMDD, labelHC] = parseLabels(label);
assert(any(labelMDD) && any(labelHC), 'Need both MDD and HC subjects.');

% ---------- Compute Cohen''s d matrices ----------
d_before = edgeCohensD(FC, labelMDD, labelHC);
d_after  = edgeCohensD(FC_combat, labelMDD, labelHC);
d_delta  = d_after - d_before;

% Mask diagonal for display (Nature-style: hide diagonal)
d_before(1:nROI1+1:end) = NaN;
d_after(1:nROI1+1:end)  = NaN;
d_delta(1:nROI1+1:end)  = NaN;

% ---------- Determine symmetric color limits ----------
maxAbs = max([abs(d_before(:)); abs(d_after(:))], [], 'omitnan');
maxAbs = max(maxAbs, eps);
clim = [-maxAbs, maxAbs];

maxAbsDelta = max(abs(d_delta(:)), [], 'omitnan');
maxAbsDelta = max(maxAbsDelta, eps);
climDelta = [-maxAbsDelta, maxAbsDelta];

% ---------- Plot (Nature-ish style) ----------
fig = figure('Color','w','Units','pixels','Position',[80 80 1350 450]);

tiledlayout(fig,1,3,'Padding','compact','TileSpacing','compact');

% Use a clean diverging colormap
cmap = blueWhiteRed(256);

% Panel 1: Before
nexttile;
imagesc(d_before); axis image tight;
set(gca,'YDir','normal');
colormap(gca,cmap); caxis(clim);
title("Before ComBat", 'FontWeight','normal');
% formatAxes(nROI1, roiNames);
cb1 = colorbar; cb1.Box = 'off'; cb1.TickDirection='out';
ylabel(cb1,"Cohen''s d (MDD - HC)");

% Panel 2: After
nexttile;
imagesc(d_after); axis image tight;
set(gca,'YDir','normal');
colormap(gca,cmap); caxis(clim);
title("After ComBat", 'FontWeight','normal');
% formatAxes(nROI1, roiNames);
cb2 = colorbar; cb2.Box = 'off'; cb2.TickDirection='out';
ylabel(cb2,"Cohen''s d (MDD - HC)");

% Panel 3: Delta
nexttile;
imagesc(d_delta); axis image tight;
set(gca,'YDir','normal');
colormap(gca,cmap); caxis(climDelta);
title("\Delta d (After - Before)", 'FontWeight','normal');
% formatAxes(nROI1, roiNames);
cb3 = colorbar; cb3.Box = 'off'; cb3.TickDirection='out';
ylabel(cb3,"\Delta Cohen''s d");

% Global formatting
set(findall(fig,'-property','FontName'),'FontName','Arial');
set(findall(fig,'-property','FontSize'),'FontSize',9);

% Optional: annotation labels a/b/c like Nature
addPanelLabel(fig, "a", [0.01 0.96 0.02 0.02]);
addPanelLabel(fig, "b", [0.345 0.96 0.02 0.02]);
addPanelLabel(fig, "c", [0.68 0.96 0.02 0.02]);

% ---------- Export ----------
pdfFile  = outPrefix + ".pdf";
tifFile  = outPrefix + ".tif";

exportgraphics(fig, pdfFile, 'ContentType','vector');             % journal-friendly
exportgraphics(fig, tifFile, 'Resolution',600);                   % 600 dpi TIFF

fprintf('Saved: %s\nSaved: %s\n', pdfFile, tifFile);

end

% ================== Helper functions ==================

function [idxMDD, idxHC] = parseLabels(label)
% Accept numeric (0/1), logical, categorical, string, cellstr.
if isnumeric(label) || islogical(label)
    idxMDD = (label(:) == 1);
    idxHC  = (label(:) == 0);
else
    lab = categorical(label);
    % Try common names
    cats = categories(lab);
    % Map by contains
    mddCat = cats(contains(lower(cats),'mdd') | contains(lower(cats),'depress'));
    hcCat  = cats(contains(lower(cats),'hc')  | contains(lower(cats),'control'));
    if ~isempty(mddCat) && ~isempty(hcCat)
        idxMDD = (lab == mddCat{1});
        idxHC  = (lab == hcCat{1});
    else
        % Fallback: assume first category is HC, second is MDD (edit if needed)
        warning('Could not auto-detect label names; using category order fallback.');
        idxHC  = (lab == cats{1});
        idxMDD = (lab == cats{2});
    end
end
end

function dMat = edgeCohensD(FC3, idxMDD, idxHC)
% Compute Cohen's d for each edge (ROI,ROI) between MDD and HC.
% FC3: nROI x nROI x nSub
X1 = FC3(:,:,idxMDD);  % MDD
X0 = FC3(:,:,idxHC);   % HC

m1 = mean(X1,3,'omitnan');
m0 = mean(X0,3,'omitnan');

s1 = std(X1,0,3,'omitnan');
s0 = std(X0,0,3,'omitnan');

n1 = sum(idxMDD);
n0 = sum(idxHC);

sp = sqrt(((n1-1).*s1.^2 + (n0-1).*s0.^2) ./ max((n1+n0-2),1));
dMat = (m1 - m0) ./ sp;

% Handle zero-variance edges
dMat(~isfinite(dMat)) = 0;

% Optional: Hedges' g correction (uncomment if you want small-sample correction)
% J = 1 - 3/(4*(n1+n0) - 9);
% dMat = J * dMat;
end

function formatAxes(nROI, roiNames)
% Minimal ticks for readability. If ROI count is large, hide tick labels.
ax = gca;
ax.TickDirection = 'out';
ax.Box = 'off';
ax.LineWidth = 0.75;

if ~isempty(roiNames) && numel(roiNames)==nROI && nROI<=80
    ax.XTick = 1:nROI; ax.YTick = 1:nROI;
    ax.XTickLabel = roiNames; ax.YTickLabel = roiNames;
    ax.XTickLabelRotation = 90;
else
    % show only a few ticks
    nt = 8;
    ticks = round(linspace(1,nROI,nt));
    ax.XTick = ticks; ax.YTick = ticks;
    ax.XTickLabel = string(ticks);
    ax.YTickLabel = string(ticks);
end
xlabel('ROI'); ylabel('ROI');
end

function cmap = blueWhiteRed(m)
% Simple diverging colormap (blue -> white -> red).
if nargin < 1, m = 256; end
bottom = [0 0.2 0.8];
middle = [1 1 1];
top    = [0.8 0 0];

n1 = floor(m/2);
n2 = m - n1;

cmap1 = [linspace(bottom(1),middle(1),n1)', ...
         linspace(bottom(2),middle(2),n1)', ...
         linspace(bottom(3),middle(3),n1)'];
cmap2 = [linspace(middle(1),top(1),n2)', ...
         linspace(middle(2),top(2),n2)', ...
         linspace(middle(3),top(3),n2)'];
cmap = [cmap1; cmap2];
end

function addPanelLabel(fig, txt, pos)
% pos is [x y w h] in normalized figure units
annotation(fig,'textbox',pos,'String',txt,'LineStyle','none', ...
    'FontName','Arial','FontSize',11,'FontWeight','bold', ...
    'HorizontalAlignment','left','VerticalAlignment','top');
end
