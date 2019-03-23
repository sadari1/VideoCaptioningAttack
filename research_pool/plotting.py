import numpy as np
import os
import scipy
import scipy.stats
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import pickle as pkl
import skvideo.io

from mpl_toolkits.mplot3d import Axes3D

GOLDEN_RATIO = 1.618

#onecolumn_width = 3.487
# onecolumn_width = 3.335 # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335
onecolumn_width = 3.00 # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335
#onecolumn_width2 = 3.335 # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335
onecolumn_height = onecolumn_width / GOLDEN_RATIO
#onecolumn_height2 = onecolumn_width2 / GOLDEN_RATIO # usenixstylefile lists textwidth as 7in, with colsep 0.33 (7-0.33)/2 = 3.335

twocolumn_width = onecolumn_width * 2
twocolumn_height = (twocolumn_width / GOLDEN_RATIO) - 1
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

    # pkl_path =
    # video_path =


def comparison_plot(batched_t, original, name):
    n_col = 4
    n_rows = 2

    original = np.array(original)

    for i in range(n_rows):
        fig, axes = plt.subplots(nrows=1, ncols=n_col, sharey=True, sharex=True, figsize=[24, 6])

        k = 0
        for j in range(n_col):

            # Plot original first
            if i is 0:
                typ = 'orig'
                showing = original[k]

                if batched_t.shape[-1] == 3:
                    showing = showing
                else:
                    showing = showing.permute(1, 2, 0)

            # Plot adversarial below it
            else:
                typ = 'adv'
                showing = batched_t[k]

                if batched_t.shape[-1] == 3:
                    showing = showing
                else:
                    showing = showing.permute(1, 2, 0)

            axes[j].imshow(showing, aspect='equal')
            axes[j].set_xticks([])
            axes[j].set_yticks([])
            axes[j].set_xticklabels([])
            axes[j].set_yticklabels([])

            k += 1

        # 0.8 and 0.5 =y for pineapple,
        # plt.figtext(0.5, 0.96, "Original Frames", va="center", ha="center", size=12)
        # plt.figtext(0.5, 0.5, "Adversarial Frames", va="center", ha="center", size=12)
        fig.set_size_inches(twocolumn_width, twocolumn_height_half)
        # plt.tight_layout()
        fig.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0, hspace=0)

        # fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.savefig(os.path.join(out_dir, '{}_{}.pdf'.format(name, typ)), bbox_inches='tight')


def sub_exp():
    c_vals = [0.01]#, 0.1, 1, 10, 100]
    for c in c_vals:
        pkl_path = '/media/wgar/WD_Passport_4TB1/data/VideoCaptioningAttack/032119-214535/{}'.format(c)
        video_path = '/media/wgar/WD_Passport_4TB1/dataset/MSVD/YouTubeClips/'

        path_list = [os.path.join(root, name)
                     for root, dirs, files in os.walk(pkl_path)
                     for name in files
                     if name.endswith((".pkl"))]
        print("\n", len(path_list))
        for path in path_list:
            split_path = path.split('/')
            split_filename = split_path[-1].split('.')[0]
            name = split_filename.replace('_adversarial', '')

            with open(path, 'rb') as f:
                print("Loaded %s" % path)
                x = pkl.load(f)

            originalvid_path = video_path + name + ".avi"
            original = skvideo.io.vread(originalvid_path)[0:4]
            # pass_in, iterations, delta, original_caption, target_caption
            img = x['pass_in']
            name += 'c_value_{}'.format(c)

            #   delta = x['delta']
            #   print(img.shape)
            #   print("{}\nOriginal caption: {}\nTarget caption: {}".format(x['iterations'], x['original_caption'], x['target_caption']))
            comparison_plot(img / 255., original, name)

        # plot_frames(x['pass_in']/255.)

        # plot_frames(delta*255)


if __name__ == '__main__':
    # sample_plotting()
    sub_exp()
