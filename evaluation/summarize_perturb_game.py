import argparse, os
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
import utils
from evaluation.measures import TrendRecord
from utils import log
import matplotlib
from pathlib import Path

FONTSIZE = 14
AXIS_FONTSIZE = 12
plt.rc('text', usetex=True), plt.rc('font', size=FONTSIZE)
plt.rc('xtick', labelsize=AXIS_FONTSIZE), plt.rc('ytick', labelsize=AXIS_FONTSIZE)
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

EVAL_MODES = ['ins', 'del']
SIGLIP_MIN_NORM_FACTOR = 1e-2
STYLE_CODE = {'ins': '-', 'del': '--'}
EXPL_NAME_MAPPING = {
    'gefi': 'GEFI',
    'fixlip': 'FIxLIP',
    'faithshap': 'Faith-Shap',
    'exclip': 'exCLIP',
    'proxyspex': 'ProxySPEX',
    'random': 'Random'
}

def load_records(root, filenames, topk=5):
    """
    The trend files are saved as dictionaries indexed by token positions.
    Each dictionary entry contains two trends, each corresponding to model responses
    to one entry in a contrastive pair.
    """
    top_trends = [TrendRecord() for _ in range(topk)]
    for fn in filenames:
        fpath = f'{root}/{fn}'
        trends = pickle.load(open(fpath, 'rb'))
        for recorder, tkn_idx in zip(top_trends, list(trends.keys())[:topk]):
            recorder.add_entry(*trends[tkn_idx])  # Shape of trends[tkn_idx]: (2, num_queries)
    return top_trends 

def get_plot_filling(means, stds, upper_bound=1000, lower_bound=-1000):
    upper = np.clip(np.array(means) + np.array(stds), lower_bound, upper_bound) 
    lower = np.clip(np.array(means) - np.array(stds), lower_bound, upper_bound) 
    return upper, lower

def plot_with_std(ax, means, stds, lbl='', enable_std=True, c=None, linestyle='-'):
    """
    NOTE: line plot of means, with surrounding region filled by stds.
    """
    steps = len(means)
    x_axis = [i for i in range(1, steps+1)]

    upper, lower = get_plot_filling(means, stds)

    ax.plot(x_axis, means, label=lbl, c=c, linestyle=linestyle)
    if enable_std:
        ax.fill_between(x_axis, upper, lower, alpha=.35, color=c)


def main(args):

    tested_explainers = ['gefi', 'exclip', 'faithshap', 'fixlip', 'proxyspex', 'random']
    # tested_explainers = ['gefir', 'gefirsc', 'gefirv', 'gefib', 'gefivr']
    trends_by_expl = dict()
    num_keys = args.num_key_idx

    # Visualize perturbation curves if option specified
    flag_visualize = args.flag_visualize
    min_norm_factor = SIGLIP_MIN_NORM_FACTOR if 'siglip' in args.model.lower() or 'RN50x4' in args.model else None

    for expl_name in tested_explainers:
        log(f'======== {EXPL_NAME_MAPPING[expl_name]} ========')

        flag_missing = False
        # Load changing trends for two evaluation modes for later summarizing
        for eval_mode in EVAL_MODES:
            root_trd_path = str(Path(utils.get_file_path(args.root_result_path, args.model, args.dataset, f"{expl_name}/trend_{eval_mode}", "*", "*")).parent)
            if not os.path.exists(root_trd_path):
                log(f'Missing folder: [{root_trd_path}]')
                flag_missing = True
                continue
            # Ensure that the directory contains EXCLUSIVELY the trend records of interest!
            filenames = [fn for fn in os.listdir(root_trd_path) if '.pkl' in fn]
            top_trends = load_records(root_trd_path, filenames)

            trends_by_expl[f'{expl_name}_{eval_mode}'] = top_trends

        # Skip explainer with incomplete records
        if flag_missing:
            print('')
            continue

        # Area-between-perturbation-curves + Visualization
        trends_pack = zip(
            trends_by_expl[f'{expl_name}_{EVAL_MODES[0]}'], # ins
            trends_by_expl[f'{expl_name}_{EVAL_MODES[1]}'], # del
        )

        if flag_visualize:
            fig, axs = plt.subplots(1, num_keys, figsize=(4*num_keys, 4))
            suptitle = EXPL_NAME_MAPPING[expl_name]
            fig.suptitle(f'Explainer: {suptitle}', y=1.01)
            axs[0].set_ylabel(r'$\Delta f~(\%)$')

        abpc_del_list, abpc_ins_list, diff_abpc_list = [], [], []
        abpc_del_samples, abpc_ins_samples, diff_abpc_samples = [], [], []
        for i, (ins_trend, del_trend) in enumerate(trends_pack):  # For each of the topk anchors

            if len(del_trend.w_anchor) == 0 or len(ins_trend.w_anchor) == 0:
                log('Skipping empty trend record')
                continue

            del_avg_w, del_avg_wo = del_trend.get_per_sample_summary()  # Shape of each: (num_queries,)
            ins_avg_w, ins_avg_wo = ins_trend.get_per_sample_summary()
            del_norm_factor = torch.vstack(del_trend.w_anchor)[:, -1].abs()
            ins_norm_factor = torch.vstack(ins_trend.w_anchor)[:, -1].abs()
            valid_samples = (del_norm_factor != 0) & (ins_norm_factor != 0)
            if min_norm_factor is not None:
                valid_samples = valid_samples & (del_norm_factor >= min_norm_factor) & (ins_norm_factor >= min_norm_factor)
                # num_filtered = valid_samples.numel() - valid_samples.sum().item()
                # if num_filtered > 0:
                    # log(f'Filtered {num_filtered}/{valid_samples.numel()} samples with norm_factor < {min_norm_factor:g}')
            if valid_samples.sum() == 0:
                log(f'Skipping Top-{i+1} anchors: all samples have invalid norm_factor')
                continue
            del_avg_w, del_avg_wo = del_avg_w[valid_samples], del_avg_wo[valid_samples]
            ins_avg_w, ins_avg_wo = ins_avg_w[valid_samples], ins_avg_wo[valid_samples]
            del_diff = del_avg_w - del_avg_wo
            ins_diff = ins_avg_w - ins_avg_wo
            diff_in_areas = ins_diff - del_diff

            if i == 0:
                full_degrad = del_avg_w[-1] - del_avg_w[0]

            del_norm_abpc = del_diff.mean() 
            ins_norm_abpc = ins_diff.mean() 
            diff_abpc = diff_in_areas.mean()
            del_norm_std = del_diff.std() 
            ins_norm_std = ins_diff.std() 
            diff_abpc_std = diff_in_areas.std()

            # diff_in_areas = ins_norm_abpc - del_norm_abpc

            spaces = ' ' * 5
            msg_del = f'ABPC-DEL:{del_norm_abpc:>6.4f} \u00B1 {del_norm_std:>4.2f} '  # Lower is better
            msg_ins = f'ABPC-INS:{ins_norm_abpc:>6.4f} \u00B1 {ins_norm_std:>4.2f}'  # Higher is better
            msg_overall = f'Diff-ABPC:{diff_abpc:>6.4f} \u00B1 {diff_abpc_std:>4.2f}'  # Higher is better
            msgs = [f'[Top-{i+1} Anchors]', msg_del, msg_ins, msg_overall]
            msg = spaces.join(msgs)
            log(msg)

            abpc_del_list.append(del_norm_abpc), abpc_ins_list.append(ins_norm_abpc), diff_abpc_list.append(diff_abpc)
            abpc_del_samples.append(del_diff)
            abpc_ins_samples.append(ins_diff)
            diff_abpc_samples.append(diff_in_areas)

            if flag_visualize:
                ax = axs[i]

                if i == 0:
                    y_ticks = np.linspace(0, full_degrad, 6)
                    y_ticklabels = np.linspace(0, -100, 6, dtype=int)
                    ax.set_yticks(y_ticks), ax.set_yticklabels(y_ticklabels)
                    ax.yaxis.set_label_coords(-0.15, 0.5)
                else:
                    ax.sharey(ax)
                    ax.tick_params(axis="y", left=False, labelleft=False)

                per_step_del_avg_w, per_step_del_avg_wo = del_trend.get_summary()  # Shape of each: (num_queries,)
                per_step_ins_avg_w, per_step_ins_avg_wo = ins_trend.get_summary()
                ax.plot(per_step_del_avg_w, linestyle='--', c='royalblue'), 
                ax.plot(per_step_del_avg_wo, linestyle='--', c='indianred')
                ax.plot(per_step_ins_avg_w, linestyle='-', c='royalblue'), 
                ax.plot(per_step_ins_avg_wo, linestyle='-', c='indianred')

                x_ticks = np.arange(len(per_step_del_avg_w))
                alpha = 0.10
                ax.fill_between(x_ticks, per_step_del_avg_w, per_step_ins_avg_w, color='royalblue', alpha=alpha)
                ax.fill_between(x_ticks, per_step_del_avg_wo, per_step_ins_avg_wo, color='indianred', alpha=alpha)

                del_abpc_as_str = f'{del_norm_abpc:.4f}'
                ins_abpc_as_str = f'{ins_norm_abpc:.4f}'

                ax.text(0.03, 0.12, 'ABPC-DEL:',
                            transform=axs[i].transAxes, ha='left', va='top')
                ax.text(0.55, 0.12, f'{del_abpc_as_str}',
                            transform=axs[i].transAxes, ha='right', va='top')
                ax.text(0.03, 0.06, 'ABPC-INS:',
                            transform=axs[i].transAxes, ha='left', va='top')
                ax.text(0.55, 0.06, f'{ins_abpc_as_str}',
                            transform=axs[i].transAxes, ha='right', va='top')

                ax.set_title(f'Avg. over Top-{i+1} Anchors')

        abpc_del_all = torch.concat(abpc_del_samples)
        abpc_ins_all = torch.concat(abpc_ins_samples)
        diff_abpc_all = torch.concat(diff_abpc_samples)
        abpc_del_mean, abpc_del_std = abpc_del_all.mean().item(), abpc_del_all.std().item()
        abpc_ins_mean, abpc_ins_std = abpc_ins_all.mean().item(), abpc_ins_all.std().item()
        diff_abpc_mean, diff_abpc_std = diff_abpc_all.mean().item(), diff_abpc_all.std().item()

        log(
            f'Average ABPC over Top-{num_keys} Anchors: '
            f'DEL: {abpc_del_mean:.4f} ± {abpc_del_std:.2f} | '
            f'INS: {abpc_ins_mean:.4f} ± {abpc_ins_std:.2f} | '
            f'Diff: {diff_abpc_mean:.4f} ± {diff_abpc_std:.2f}\n'
        )

        if flag_visualize:
            fig.subplots_adjust(wspace=0.05)
            visual_sav_path = str(Path(root_trd_path).parent / 'perturbation_curves.pdf')
            log(f'Saving plotted curves to [{visual_sav_path}]')
            fig.savefig(visual_sav_path)


if __name__ == '__main__':

    config_pth = 'configs/explain_eval_config.yaml'
    config = utils.load_config(config_pth)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    ns = argparse.Namespace()
    ns.__dict__.update(config)
    for k in ns.__dict__:
        parser.add_argument('--'+k, type=type(getattr(ns, k)))
    _args = parser.parse_known_args(namespace=ns)[0]

    log(_args)

    main(_args)