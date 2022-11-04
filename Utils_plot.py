import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#ff506e', '#69005f', 'tab:gray', '#0065bd', 'tab:olive', 'peru'])



def add_label(violin, label, labels):
    '''Function for adding labels to violin plots'''

    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.03, fs=None, maxasterix=None, col='dimgrey'):
    """
    Annotate barplot with p-values.
    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    :param col: color of the asterixes or text. Optional. The default is dimgrey.
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+ barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh - 0.035 * barh)

    plt.plot(barx, bary, c='dimgrey')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs, c=col)


def show_stars(p_cor, ind, bars, heights, arange_dh=False, dh=.1, flexible_dh=False, col='dimgrey'):
    '''
    Function to show brackets with asterisks indicating statistical
    significance
    Parameters
    ----------
    p_cor : array or list
        corrected p-values.
    ind : list of lists
        indices for the comparisons corresponding to the individual p-values.
    bars : list or array
        x position of boxplots.
    heights : list or array
        maximal value visualised in boxplots.
    arange_dh : float, optional
        offset above heights. The default is False.
    col : str
        color of the asterixes. Optional, the default is dimgrey.
    Returns
    -------
    int
        returns 0, when completed..
    '''

    star = p_cor < 0.05
    starstar = p_cor < 0.001

    all_nrs = []
    # dh = .1
    if arange_dh == True:
        dhs = np.array([.1, .1, .2, .1, .1, .2, .1, .1, .2, .2, .2, .2, .2, .1, .1, .1, .1, .2, .3, .1, .2, .3, .2, .2])

    fl_dh = .03
    for i in range(0, len(p_cor)):
        nr = ind[i]
        all_nrs.append(nr)
        if arange_dh:
            dh = dhs[i]
        if flexible_dh:
            dh = fl_dh
        if starstar[i] == True:
            barplot_annotate_brackets(nr[0], nr[1], '**', bars, heights, dh=dh, col=col)
            fl_dh += 0.05
        else:
            if star[i] == True:
                barplot_annotate_brackets(nr[0], nr[1], '*', bars, heights, dh=dh, col=col)
                fl_dh += 0.05

    return 0
    