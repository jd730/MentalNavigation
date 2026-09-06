% Minimal runner for mnav xcorr comparison (plot only, no unrelated blocks)


clear; clc;


script_dir = fileparts(mfilename('fullpath'));


model_mnav_dir_1 = fullfile(script_dir,'mat_files_smoothed'); %fullfile(script_dir,'mat_files');
model_mnav_dir_2 = fullfile(script_dir,'mat_files_smoothed');


maxlag = 20;
xxi = -maxlag:maxlag;
cond = 2;
seed = 43;
use_xcov = true; % false: xcorr(...,'coeff'), true: xcov(...,'coeff')
use_topk_dist = false; % true: use top-K base_hidden neurons by dist_classifier.weight
top_k_dist = 64;


[xcorr_pair_gs_1, xcorr_pair_dist_1, xcorr_pair_action_1, name_1] = compute_xcorr_from_dir(model_mnav_dir_1, seed, cond, maxlag, use_xcov, use_topk_dist, top_k_dist);
[xcorr_pair_gs_2, xcorr_pair_dist_2, xcorr_pair_action_2, name_2] = compute_xcorr_from_dir(model_mnav_dir_2, seed, cond, maxlag, use_xcov, use_topk_dist, top_k_dist);


if use_xcov
    corr_mode = 'xcov coeff';
else
    corr_mode = 'xcorr coeff';
end


if use_topk_dist
    dist_mode = sprintf('dist topK=%d by dist_classifier.weight', top_k_dist);
else
    dist_mode = 'dist all neurons';
end


hf = figure('Position',[404 78 700 788]);


[~, maxidx_gs_1] = max(xcorr_pair_gs_1, [], 2);
[~, sortid_gs_1] = sort(maxidx_gs_1);
subplot(3,2,1);
imagesc(xxi, 1:length(sortid_gs_1), xcorr_pair_gs_1(sortid_gs_1,:));
xlim([-maxlag maxlag]);
xline(0, 'r-');
xline(12);
xline(-12);
xlabel('peak phase'); ylabel('#all cell pairs');
title('grid cells mnav | mat\_files');
caxis([-1 1]);
colorbar;


[~, maxidx_gs_2] = max(xcorr_pair_gs_2, [], 2);
[~, sortid_gs_2] = sort(maxidx_gs_2);
subplot(3,2,2);
imagesc(xxi, 1:length(sortid_gs_2), xcorr_pair_gs_2(sortid_gs_2,:));
xlim([-maxlag maxlag]);
xline(0, 'r-');
xline(12);
xline(-12);
xlabel('peak phase'); ylabel('#all cell pairs');
title('grid cells mnav | mat\_files\_smoothed');
caxis([-1 1]);
colorbar;


[~, maxidx_dist_1] = max(xcorr_pair_dist_1, [], 2);
[~, sortid_dist_1] = sort(maxidx_dist_1);
subplot(3,2,3);
imagesc(xxi, 1:length(sortid_dist_1), xcorr_pair_dist_1(sortid_dist_1,:));
xlim([-maxlag maxlag]);
xline(0, 'r-');
xline(12);
xline(-12);
xlabel('peak phase'); ylabel('#all cell pairs');
title('distance mnav | mat\_files');
caxis([-1 1]);
colorbar;


[~, maxidx_dist_2] = max(xcorr_pair_dist_2, [], 2);
[~, sortid_dist_2] = sort(maxidx_dist_2);
subplot(3,2,4);
imagesc(xxi, 1:length(sortid_dist_2), xcorr_pair_dist_2(sortid_dist_2,:));
xlim([-maxlag maxlag]);
xline(0, 'r-');
xline(12);
xline(-12);
xlabel('peak phase'); ylabel('#all cell pairs');
title('distance mnav | mat\_files\_smoothed');
caxis([-1 1]);
colorbar;


[~, maxidx_action_1] = max(xcorr_pair_action_1, [], 2);
[~, sortid_action_1] = sort(maxidx_action_1);
subplot(3,2,5);
imagesc(xxi, 1:length(sortid_action_1), xcorr_pair_action_1(sortid_action_1,:));
xlim([-maxlag maxlag]);
xline(0, 'r-');
xlabel('peak phase'); ylabel('#all cell pairs');
title('action mnav | mat\_files');
caxis([-1 1]);
colorbar;


[~, maxidx_action_2] = max(xcorr_pair_action_2, [], 2);
[~, sortid_action_2] = sort(maxidx_action_2);
subplot(3,2,6);
imagesc(xxi, 1:length(sortid_action_2), xcorr_pair_action_2(sortid_action_2,:));
xlim([-maxlag maxlag]);
xline(0, 'r-');
xlabel('peak phase'); ylabel('#all cell pairs');
title('action mnav | mat\_files\_smoothed');
caxis([-1 1]);
colorbar;


sgtitle(sprintf('mnav %s comparison | %s | seed=%d | cond=%d\nleft=%s | right=%s', corr_mode, dist_mode, seed, cond, name_1, name_2), 'Interpreter', 'none');


disp('Visualization complete (no MAT file saved).');




function [xcorr_pair_gs_mnav, xcorr_pair_dist_mnav, xcorr_pair_action_mnav, selected_name] = compute_xcorr_from_dir(model_mnav_dir, seed, cond, maxlag, use_xcov, use_topk_dist, top_k_dist)
filesmnav = dir(fullfile(model_mnav_dir, '*.mat'));
if isempty(filesmnav)
    error('Input directory is empty: %s', model_mnav_dir);
end


seed_token = ['S' num2str(seed)];
idx_m = find(contains({filesmnav.name}, seed_token), 1, 'first');
if isempty(idx_m)
    parsed_seeds = nan(1, numel(filesmnav));
    for fi = 1:numel(filesmnav)
        tok = regexp(filesmnav(fi).name, 'S(\d+)Ep', 'tokens', 'once');
        if ~isempty(tok)
            parsed_seeds(fi) = str2double(tok{1});
        end
    end


    valid_idx = find(~isnan(parsed_seeds));
    if isempty(valid_idx)
        error('Requested seed S%d not found and no parseable S#Ep filenames in %s.', seed, model_mnav_dir);
    end


    [~, best_local] = min(abs(parsed_seeds(valid_idx) - seed));
    idx_m = valid_idx(best_local);
    fprintf('Requested seed S%d not found in %s. Using nearest available seed S%d (%s).\n', ...
        seed, model_mnav_dir, parsed_seeds(idx_m), filesmnav(idx_m).name);
end


selected_name = filesmnav(idx_m).name;
mnav_path = fullfile(model_mnav_dir, selected_name);
try
    mnav_data = load(mnav_path, 'hidden', 'base_hidden', 'gs');
catch ME
    if contains(model_mnav_dir, 'mat_files_smoothed')
        raw_dir = strrep(model_mnav_dir, 'mat_files_smoothed', 'mat_files');
        raw_path = fullfile(raw_dir, selected_name);
        if ~isfile(raw_path)
            rethrow(ME);
        end


        script_dir_local = fileparts(mfilename('fullpath'));
        py_script = fullfile(script_dir_local, 'smooth_gs_circular.py');
        if ~isfile(py_script)
            rethrow(ME);
        end


        run_python_smoothing(py_script, raw_path, model_mnav_dir, 1.0);
        mnav_data = load(mnav_path, 'hidden', 'base_hidden', 'gs');
        fprintf('Warning: smoothed MAT was regenerated via Python for %s\n', selected_name);
    else
        rethrow(ME);
    end
end


if ~isfield(mnav_data, 'hidden') || ~isfield(mnav_data, 'base_hidden') || ~isfield(mnav_data, 'gs')
    error('Required vars missing in %s. Expected hidden, base_hidden, gs.', mnav_path);
end


action_units_mnav = squeeze(mnav_data.hidden(cond,:,:));
dist_units_mnav = squeeze(mnav_data.base_hidden(cond,:,:));
gs_units_mnav = squeeze(mnav_data.gs(cond,:,:));


gs_units_mnav(isnan(gs_units_mnav)) = 0;
action_units_mnav(isnan(action_units_mnav)) = 0;
dist_units_mnav(isnan(dist_units_mnav)) = 0;


if use_topk_dist
    dist_weight = load_dist_classifier_weight(mnav_path, size(dist_units_mnav,2));
    k = min(top_k_dist, numel(dist_weight));
    [~, order] = sort(abs(dist_weight(:)), 'descend');
    top_idx = order(1:k);
    dist_units_mnav = dist_units_mnav(:, top_idx);
end


nn_xcorr = size(dist_units_mnav,2);
ii = 1;
for nn = 1:nn_xcorr
    for mm = (nn+1):nn_xcorr
        if use_xcov
            xcorr_pair_action_mnav(ii,:) = xcov(action_units_mnav(:,nn), action_units_mnav(:,mm), maxlag, 'coeff');
            xcorr_pair_dist_mnav(ii,:) = xcov(dist_units_mnav(:,nn), dist_units_mnav(:,mm), maxlag, 'coeff');
        else
            xcorr_pair_action_mnav(ii,:) = xcorr(action_units_mnav(:,nn), action_units_mnav(:,mm), maxlag, 'coeff');
            xcorr_pair_dist_mnav(ii,:) = xcorr(dist_units_mnav(:,nn), dist_units_mnav(:,mm), maxlag, 'coeff');
        end
        ii = ii + 1;
    end
end


nn_xcorr = size(gs_units_mnav,2);
ii = 1;
for nn = 1:nn_xcorr
    for mm = (nn+1):nn_xcorr
        if use_xcov
            xcorr_pair_gs_mnav(ii,:) = xcov(gs_units_mnav(30:50,nn), gs_units_mnav(30:50,mm), maxlag, 'coeff');
        else
            xcorr_pair_gs_mnav(ii,:) = xcorr(gs_units_mnav(30:50,nn), gs_units_mnav(30:50,mm), maxlag, 'coeff');
        end
        ii = ii + 1;
    end
end
end




function run_python_smoothing(py_script, input_file, output_dir, sigma)
py_exes = {'/Users/jdhwang/anaconda3/bin/python', 'python3', 'python', '/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3'};
status = -1;
last_out = '';
for ei = 1:numel(py_exes)
    exe = py_exes{ei};
    cmd = sprintf('"%s" "%s" --input-file "%s" --output-dir "%s" --sigma %.6f', ...
        exe, py_script, input_file, output_dir, sigma);
    [status_i, out_i] = system(cmd);
    if status_i == 0
        status = 0;
        fprintf('Python smoothing succeeded with %s\n', exe);
        return;
    end
    last_out = out_i;
end


error('Python smoothing failed for all candidates. Last output:\n%s', last_out);
end




function dist_weight = load_dist_classifier_weight(mnav_path, num_units)
dist_weight = [];


if ~isfile(mnav_path)
    error('MAT file does not exist: %s', mnav_path);
end


% Try MATLAB-friendly variable names first (if present).
vars = who('-file', mnav_path);
if any(strcmp(vars, 'dist_classifier_weight'))
    temp = load(mnav_path, 'dist_classifier_weight');
    dist_weight = temp.dist_classifier_weight;
elseif any(strcmp(vars, 'dist_classifierWeight'))
    temp = load(mnav_path, 'dist_classifierWeight');
    dist_weight = temp.dist_classifierWeight;
end


% Fallback: load invalid MATLAB key names through external python scipy.io.loadmat.
if isempty(dist_weight)
    try
        py_exes = {'/Users/jdhwang/anaconda3/bin/python', 'python3', 'python', '/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3'};
        status = -1;
        out = '';
        used_exe = '';
        py_code = [ ...
            'import scipy.io, numpy as np, json, sys; ' ...
            'p=sys.argv[1]; ' ...
            'm=scipy.io.loadmat(p); ' ...
            'w=m.get(''''dist_classifier.weight'''', m.get(''''dist_classifier_weight'''', None)); ' ...
            'print(''''__MISSING__'''' if w is None else json.dumps(np.ravel(w).tolist()))' ...
        ];
        for ei = 1:numel(py_exes)
            exe = py_exes{ei};
            py_cmd = sprintf([
                '"%s" -c "%s" "%s"'
                ], exe, py_code, mnav_path);


            [status_i, out_i] = system(py_cmd);
            if status_i == 0
                status = status_i;
                out = out_i;
                used_exe = exe;
                break;
            end
        end


        if status ~= 0
            error('External python command failed for all candidates: %s', strjoin(py_exes, ', '));
        end


        out = strtrim(out);
        if strcmp(out, '__MISSING__')
            error('dist_classifier.weight key not found via external python in %s', mnav_path);
        end


        dist_weight = double(jsondecode(out));
        disp(['Loaded dist_classifier.weight using ' used_exe]);
    catch ME
        error(['Could not load dist_classifier.weight from ' mnav_path '. ' ...
               'Ensure one of python3/python paths has scipy installed, or add a MATLAB-safe variable dist_classifier_weight. ' ...
               'Original error: ' ME.message]);
    end
end


dist_weight = dist_weight(:)';
if numel(dist_weight) ~= num_units
    error('dist_classifier.weight size mismatch in %s (weight=%d, base_hidden units=%d).', mnav_path, numel(dist_weight), num_units);
end
end

