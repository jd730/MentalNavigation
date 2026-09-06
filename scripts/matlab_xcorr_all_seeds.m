% Batch runner: iterate every seed under mat_files_smoothed/ (linked to our
% ReLU mat_exports/), run compute_xcorr_from_dir(seed, cond=2), and save one
% PDF per (model, seed) into results/ReLU/xcorr/<model>/.
%
% draw_xcorr.m compares two source dirs side-by-side; here we run a single
% source so the layout is 3x1 (grid cells / distance RNN / action RNN).

script_dir = fileparts(mfilename('fullpath'));

% analyze_relu.py's xcorr stage sets XCORR_MAT_DIR and XCORR_OUT_ROOT before
% invoking us. Fall back to the symlinked defaults for standalone runs.
env_mat = getenv('XCORR_MAT_DIR');
if isempty(env_mat)
    mat_dir = fullfile(script_dir, 'mat_files_smoothed');
else
    mat_dir = env_mat;
end
env_out = getenv('XCORR_OUT_ROOT');
if isempty(env_out)
    out_root = '/home/jdhwang/mental_navigation/results/ReLU/xcorr';
else
    out_root = env_out;
end

maxlag = 20;
xxi = -maxlag:maxlag;
cond = 2;
use_xcov = true;

files = dir(fullfile(mat_dir, '*.mat'));
for k = 1:numel(files)
    name = files(k).name;
    tok  = regexp(name, 'S(\d+)Ep', 'tokens', 'once');
    if isempty(tok)
        continue;
    end
    seed = str2double(tok{1});
    if startsWith(name, 'VHA-ReLU-D_')
        model = 'VHA-ReLU-D';
    elseif startsWith(name, 'VHA-ReLU_')
        model = 'VHA-ReLU';
    else
        continue;
    end

    fprintf('%s seed %d (%s)\n', model, seed, name);
    try
        [xg, xd, xa, ~] = compute_xcorr_from_file( ...
            fullfile(mat_dir, name), cond, maxlag, use_xcov, false, 64);
    catch ME
        fprintf('  seed %d failed: %s\n', seed, ME.message);
        continue;
    end

    % Taller figure + pbaspect per subplot so all three panels have the same
    % visual aspect regardless of pair count (36 gs pairs vs 32k RNN pairs).
    % Without pbaspect, imagesc + `axis normal` stretches each panel to fill
    % its subplot rectangle, so RNN panels get squished flat while gs stays
    % roughly square. Fixed pbaspect([1 1.2 1]) gives every panel the same
    % slightly-taller-than-wide box (~1.2:1 tall:wide).
    hf = figure('Position', [100 100 600 1400], 'Visible', 'off');
    n_layers = 2;                          % gs + action; add dist if present
    if ~isempty(xd)
        n_layers = 3;
    end
    row = 1;

    % gs (grid cells / EC)
    [~, maxidx] = max(xg, [], 2);
    [~, sortid] = sort(maxidx);
    subplot(n_layers, 1, row); row = row + 1;
    imagesc(xxi, 1:size(xg, 1), xg(sortid, :));
    xlim([-maxlag maxlag]); xline(0, 'r-'); xline(12); xline(-12);
    xlabel('peak phase'); ylabel('# cell pairs');
    title('grid cells (EC)'); caxis([-1 1]); colorbar;
    pbaspect([1 1.2 1]);

    % dist RNN (double-CTRNN only)
    if ~isempty(xd)
        [~, maxidx] = max(xd, [], 2);
        [~, sortid] = sort(maxidx);
        subplot(n_layers, 1, row); row = row + 1;
        imagesc(xxi, 1:size(xd, 1), xd(sortid, :));
        xlim([-maxlag maxlag]); xline(0, 'r-'); xline(12); xline(-12);
        xlabel('peak phase'); ylabel('# cell pairs');
        title('distance RNN'); caxis([-1 1]); colorbar;
        pbaspect([1 1.2 1]);
    end

    % action RNN
    [~, maxidx] = max(xa, [], 2);
    [~, sortid] = sort(maxidx);
    subplot(n_layers, 1, row);
    imagesc(xxi, 1:size(xa, 1), xa(sortid, :));
    xlim([-maxlag maxlag]); xline(0, 'r-'); xline(12); xline(-12);
    xlabel('peak phase'); ylabel('# cell pairs');
    title('action RNN'); caxis([-1 1]); colorbar;
    pbaspect([1 1.2 1]);

    sgtitle(sprintf('%s  seed=%d  cond=%d  (MATLAB xcov coeff)', ...
                    model, seed, cond), 'Interpreter', 'none');

    out_dir = fullfile(out_root, model);
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    out_pdf = fullfile(out_dir, sprintf('xcorr_S%d_cond%d.pdf', seed, cond));
    exportgraphics(hf, out_pdf, 'Resolution', 150);
    close(hf);
end

fprintf('done.\n');


function [xcorr_pair_gs, xcorr_pair_dist, xcorr_pair_action, selected_name] = ...
        compute_xcorr_from_file(mat_path, cond, maxlag, use_xcov, ...
                                use_topk_dist, top_k_dist)
% Callers pass the full mat file path so that VHA-ReLU vs VHA-ReLU-D can be
% disambiguated by the outer loop (both models share the same 'S<seed>Ep'
% substring; a directory-level search would just pick whichever sorts first).
if ~isfile(mat_path); error('mat file not found: %s', mat_path); end
[~, selected_stem, selected_ext] = fileparts(mat_path);
selected_name = [selected_stem selected_ext];
% Load only the fields we need; the .mat also carries 'classifier.weight' /
% 'dist_classifier.weight' whose dots break MATLAB's struct-field naming.
% Load base_hidden inside a try since VHA-ReLU (single-CTRNN) omits it.
data = load(mat_path, 'hidden', 'gs', 'trajs', 'suc');
try
    tmp = load(mat_path, 'base_hidden');
    if isfield(tmp, 'base_hidden')
        data.base_hidden = tmp.base_hidden;
    end
catch
    % ok - single-CTRNN model, no base_hidden
end

if ~isfield(data, 'hidden'); error('missing hidden in %s', mat_path); end
if ~isfield(data, 'gs');     error('missing gs in %s', mat_path); end

action_units = squeeze(data.hidden(cond, :, :));
gs_units     = squeeze(data.gs(cond, :, :));
action_units(isnan(action_units)) = 0;
gs_units(isnan(gs_units)) = 0;

if isfield(data, 'base_hidden')
    dist_units = squeeze(data.base_hidden(cond, :, :));
    dist_units(isnan(dist_units)) = 0;
    if use_topk_dist && isfield(data, 'dist_classifier_weight')
        w = double(data.dist_classifier_weight(:));
        k = min(top_k_dist, numel(w));
        [~, ord] = sort(abs(w), 'descend');
        dist_units = dist_units(:, ord(1:k));
    end
else
    dist_units = [];
end

% action pairs
nn_x = size(action_units, 2); ii = 1;
xcorr_pair_action = zeros(nn_x * (nn_x - 1) / 2, 2 * maxlag + 1);
for nn = 1:nn_x
    for mm = (nn + 1):nn_x
        if use_xcov
            xcorr_pair_action(ii, :) = xcov(action_units(:, nn), action_units(:, mm), maxlag, 'coeff');
        else
            xcorr_pair_action(ii, :) = xcorr(action_units(:, nn), action_units(:, mm), maxlag, 'coeff');
        end
        ii = ii + 1;
    end
end

% dist pairs
if ~isempty(dist_units)
    nn_x = size(dist_units, 2); ii = 1;
    xcorr_pair_dist = zeros(nn_x * (nn_x - 1) / 2, 2 * maxlag + 1);
    for nn = 1:nn_x
        for mm = (nn + 1):nn_x
            if use_xcov
                xcorr_pair_dist(ii, :) = xcov(dist_units(:, nn), dist_units(:, mm), maxlag, 'coeff');
            else
                xcorr_pair_dist(ii, :) = xcorr(dist_units(:, nn), dist_units(:, mm), maxlag, 'coeff');
            end
            ii = ii + 1;
        end
    end
else
    xcorr_pair_dist = [];
end

% gs pairs (windowed). draw_xcorr.m hardcodes rows 30:50 but that assumed
% a dataset where trials had valid data at those timesteps. Our .mat
% exports are right-aligned with variable trial lengths, so trials with
% |d| < ~30 have NaN (-> 0) at [30:50]. Auto-detect the last 21 valid
% rows (matching the 21-row window MATLAB used) so gs gets meaningful data.
gs_raw = squeeze(data.gs(cond, :, :));   % (T, D) with NaN in padding
valid = ~any(isnan(gs_raw), 2);
last_valid  = find(valid, 1, 'last');
first_valid = max(1, last_valid - 20);   % 21-row window
gs_win_rows = first_valid:last_valid;
nn_x = size(gs_units, 2); ii = 1;
xcorr_pair_gs = zeros(nn_x * (nn_x - 1) / 2, 2 * maxlag + 1);
for nn = 1:nn_x
    for mm = (nn + 1):nn_x
        if use_xcov
            xcorr_pair_gs(ii, :) = xcov(gs_units(gs_win_rows, nn), gs_units(gs_win_rows, mm), maxlag, 'coeff');
        else
            xcorr_pair_gs(ii, :) = xcorr(gs_units(gs_win_rows, nn), gs_units(gs_win_rows, mm), maxlag, 'coeff');
        end
        ii = ii + 1;
    end
end
end
