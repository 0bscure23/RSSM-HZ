function summary = eval_reduced_matlab(pack_dir, dlpan_matlab_dir, out_dir, data_range, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut, th_values)
% Evaluate reduced-resolution pansharpening results with DLPan MATLAB metrics.
%
% Required package structure:
%   pack_dir/pred/pred_00.mat  variable: sr, H x W x C
%   pack_dir/ref/ref_00.mat    variable: gt, H x W x C
%   pack_dir/index_map.csv
%
% Example:
%   eval_reduced_matlab('.../qb_matlab_eval', '.../DLPan-Toolbox/02-Test-toolbox-for-traditional-and-DL(Matlab)', '.../matlab_metrics', 2047, 4, 11, 32, 1, 21, 1);

if nargin < 3 || isempty(out_dir), out_dir = fullfile(pack_dir, 'matlab_metrics'); end
if nargin < 4 || isempty(data_range), data_range = 2047; end
if nargin < 5 || isempty(ratio), ratio = 4; end
if nargin < 6 || isempty(L), L = 11; end
if nargin < 7 || isempty(Qblocks_size), Qblocks_size = 32; end
if nargin < 8 || isempty(flag_cut_bounds), flag_cut_bounds = 1; end
if nargin < 9 || isempty(dim_cut), dim_cut = 21; end
if nargin < 10 || isempty(th_values), th_values = 1; end

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

if exist('pkg', 'file')
    try
        pkg('load', 'image');
    catch
        % MATLAB does not have pkg; Octave without image can still use the
        % local mean2 fallback for the paper metrics.
    end
end

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(genpath(fullfile(dlpan_matlab_dir, 'Tools')));
addpath(genpath(fullfile(dlpan_matlab_dir, 'Quality_Indices')));

% DLPan's indexes_evaluation.m changes into ./Quality_Indices with a
% relative path, so call it from the toolbox root for MATLAB/Octave parity.
old_pwd = pwd;
cleanup_pwd = onCleanup(@() cd(old_pwd));
cd(dlpan_matlab_dir);

index_csv = fullfile(pack_dir, 'index_map.csv');
rows = read_index_csv(index_csv);
n = numel(rows);

per_sample = zeros(n, 7);
for k = 1:n
    idx = rows(k).index;
    pred_file = fullfile(pack_dir, rows(k).pred_mat);
    ref_file = fullfile(pack_dir, rows(k).ref_mat);

    pred = load(pred_file);
    ref = load(ref_file);
    if ~isfield(pred, 'sr'), error('Missing sr in %s', pred_file); end
    if ~isfield(ref, 'gt'), error('Missing gt in %s', ref_file); end

    I_F = double(pred.sr);
    I_GT = double(ref.gt);
    if ~isequal(size(I_F), size(I_GT))
        error('Shape mismatch at index %d: pred=%s gt=%s', idx, mat2str(size(I_F)), mat2str(size(I_GT)));
    end

    [Q_avg, SAM_idx, ERGAS_idx, SCC_idx, Q2n_idx] = indexes_evaluation( ...
        I_F, I_GT, ratio, L, Qblocks_size, flag_cut_bounds, dim_cut, th_values);
    PSNR_idx = psnr_mean_bands(I_F, I_GT, data_range, flag_cut_bounds, dim_cut, th_values, L);

    per_sample(k, :) = [idx, PSNR_idx, SAM_idx, ERGAS_idx, Q2n_idx, Q_avg, SCC_idx];
end

summary = struct();
summary.num_samples = n;
summary.PSNR = mean(per_sample(:, 2));
summary.SAM = mean(per_sample(:, 3));
summary.ERGAS = mean(per_sample(:, 4));
summary.Q2n = mean(per_sample(:, 5));
summary.Q_avg = mean(per_sample(:, 6));
summary.SCC = mean(per_sample(:, 7));
summary.data_range = data_range;
summary.ratio = ratio;
summary.L = L;
summary.Qblocks_size = Qblocks_size;
summary.flag_cut_bounds = flag_cut_bounds;
summary.dim_cut = dim_cut;
summary.th_values = th_values;

save(fullfile(out_dir, 'matlab_metrics_summary.mat'), 'summary', 'per_sample');
write_metrics_csv(fullfile(out_dir, 'matlab_metrics_per_sample.csv'), per_sample);
write_summary_txt(fullfile(out_dir, 'matlab_metrics_summary.txt'), summary);

fprintf('MATLAB-style reduced-resolution metrics\n');
fprintf('Samples: %d\n', summary.num_samples);
fprintf('PSNR: %.6f\n', summary.PSNR);
fprintf('SAM: %.6f\n', summary.SAM);
fprintf('ERGAS: %.6f\n', summary.ERGAS);
fprintf('Q2n/Q4/Q8: %.6f\n', summary.Q2n);
fprintf('Q_avg: %.6f\n', summary.Q_avg);
fprintf('SCC: %.6f\n', summary.SCC);
end

function rows = read_index_csv(index_csv)
fid = fopen(index_csv, 'r');
if fid < 0, error('Cannot open %s', index_csv); end
cleanup = onCleanup(@() fclose(fid));
header = fgetl(fid); %#ok<NASGU>
rows = struct('index', {}, 'pred_mat', {}, 'ref_mat', {});
line = fgetl(fid);
while ischar(line)
    parts = strsplit(strtrim(line), ',');
    if numel(parts) >= 3
        r.index = str2double(parts{1});
        r.pred_mat = parts{2};
        r.ref_mat = parts{3};
        rows(end + 1) = r; %#ok<AGROW>
    end
    line = fgetl(fid);
end
end

function val = psnr_mean_bands(I_F, I_GT, data_range, flag_cut_bounds, dim_cut, th_values, L)
if flag_cut_bounds
    I_GT = I_GT(dim_cut:end-dim_cut, dim_cut:end-dim_cut, :);
    I_F = I_F(dim_cut:end-dim_cut, dim_cut:end-dim_cut, :);
end
if th_values
    hi = 2^L;
    I_F(I_F > hi) = hi;
    I_F(I_F < 0) = 0;
end
bands = size(I_GT, 3);
vals = zeros(1, bands);
for b = 1:bands
    err = I_F(:, :, b) - I_GT(:, :, b);
    mse = mean(err(:) .^ 2);
    mse = max(mse, eps);
    vals(b) = 10 * log10((data_range ^ 2) / mse);
end
val = mean(vals);
end

function write_metrics_csv(path, per_sample)
fid = fopen(path, 'w');
if fid < 0, error('Cannot write %s', path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'index,PSNR,SAM,ERGAS,Q2n,Q_avg,SCC\n');
for i = 1:size(per_sample, 1)
    fprintf(fid, '%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', per_sample(i, :));
end
end

function write_summary_txt(path, summary)
fid = fopen(path, 'w');
if fid < 0, error('Cannot write %s', path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'num_samples: %d\n', summary.num_samples);
fprintf(fid, 'PSNR: %.10f\n', summary.PSNR);
fprintf(fid, 'SAM: %.10f\n', summary.SAM);
fprintf(fid, 'ERGAS: %.10f\n', summary.ERGAS);
fprintf(fid, 'Q2n_Q4_Q8: %.10f\n', summary.Q2n);
fprintf(fid, 'Q_avg: %.10f\n', summary.Q_avg);
fprintf(fid, 'SCC: %.10f\n', summary.SCC);
fprintf(fid, 'data_range: %.10f\n', summary.data_range);
fprintf(fid, 'ratio: %.10f\n', summary.ratio);
fprintf(fid, 'L: %.10f\n', summary.L);
fprintf(fid, 'Qblocks_size: %.10f\n', summary.Qblocks_size);
fprintf(fid, 'flag_cut_bounds: %.10f\n', summary.flag_cut_bounds);
fprintf(fid, 'dim_cut: %.10f\n', summary.dim_cut);
fprintf(fid, 'th_values: %.10f\n', summary.th_values);
end
