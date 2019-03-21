import numpy as np
import os
import scipy
import scipy.stats
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

GOLDEN_RATIO = 1.618

#onecolumn_width = 3.487
onecolumn_width = 3.335 # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335
#onecolumn_width2 = 3.335 # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335
onecolumn_height = onecolumn_width / GOLDEN_RATIO
#onecolumn_height2 = onecolumn_width2 / GOLDEN_RATIO # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335

twocolumn_width = onecolumn_width * 2
twocolumn_height = twocolumn_width / GOLDEN_RATIO
twocolumn_height_half = twocolumn_height / 2

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathptmx}',
                                       r'\usepackage[T1]{fontenc}',
                                       r'\usepackage[utf8]{inputenc}',
                                       r'\usepackage{pslatex}']
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('axes', labelsize=8)

out_dir = 'byproducts/'


def sample_plotting():
    """
    :return:
    """
    data = np.random.rand(10, 10)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    _x = np.arange(len(data[0]))
    _y = np.arange(len(data))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    topp = data.ravel()
    bottom = np.zeros_like(topp)
    width = depth = 0.50

    ax.bar3d(x, y, bottom, width, depth, topp, shade=True)
    ax.set_title("Super cool title")

    # fig.set_size_inches(onecolumn_width, onecolumn_height)
    fig.savefig(os.path.join(out_dir, '{}.pdf'.format(str(sample_plotting.__name__))))


if __name__ == '__main__':
    sample_plotting()
