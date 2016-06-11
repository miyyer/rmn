import csv, cPickle
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

# parse learned descriptors into a dict
def read_descriptors(desc_file):
    desc_map = {}
    f = open(desc_file, 'r')
    for i, line in enumerate(f):
        line = line.split()
        desc_map[i] = line[0]
    return desc_map

# read learned trajectories file
def read_csv(csv_file):
    reader = csv.reader(open(csv_file, 'rb'))
    all_traj = {}
    prev_book = None
    prev_c1 = None
    prev_c2 = None
    total_traj = 0
    for index, row in enumerate(reader):
        if index == 0:
            continue
        book, c1, c2 = row[:3]
        if prev_book != book or prev_c1 != c1 or prev_c2 != c2:
            prev_book = book
            prev_c1 = c1
            prev_c2 = c2
            if book not in all_traj:
                all_traj[book] = {}
            all_traj[book][c1+' AND '+c2] = []
            total_traj += 1

        else:
            all_traj[book][c1+' AND '+c2].append(array(row[4:], dtype='float32'))

    print len(all_traj), total_traj
    return all_traj

# compute locations to write labels
# only write labels when the 
def compute_centers(max_traj, smallest_shift):
    center_inds = []
    prev_topic = max_traj[0]
    tstart = 0
    for index, topic in enumerate(max_traj):
        if topic != prev_topic:
            center = int((index-tstart) / 2)
            if center > smallest_shift / 2:
                center_inds.append(tstart + center)
            tstart = index
            prev_topic = topic
    center = int((index-tstart) / 2)
    if index - tstart > smallest_shift:
        center_inds.append(tstart + center)

    return center_inds

def viz_csv(rmn_traj, rmn_descs,
    min_length=10,
    smallest_shift=1, max_viz=False,
    fig_dir=None):

    for book in rmn_traj:
        for rel in rmn_traj[book]:
            rtraj = rmn_traj[book][rel]
            if len(rtraj) > min_length and len(rtraj)<150:

                print book, rel
                plt.close()
                rtraj_mat = array(rtraj)

                if max_viz:
                    plt.title(book + ': ' + rel)
                    plt.axis('off')

                    max_rtraj = argmax(rtraj_mat, axis=1)
                    rcenter_inds = compute_centers(max_rtraj, smallest_shift)


                    for ind in range(0, len(max_rtraj)):
                        topic = max_rtraj[ind]
                        plt.axhspan(ind, ind+1, 0.2, 0.4, color=color_list[topic])

                        if ind in rcenter_inds:
                            loc = (0.43, ind + 0.5)
                            plt.annotate(rmn_descs[topic], loc, size=15,
                                verticalalignment='center',
                                color=color_list[topic])


                    plt.xlim(0, 1.0)
                    plt.arrow(0.1,0,0.0,len(rtraj),
                            head_width=0.1, head_length=len(rtraj)/12, lw=3, 
                            length_includes_head=True, fc='k', ec='k')

                    props = {'ha': 'left', 'va': 'bottom',}
                    plt.text(0.0, len(rtraj) / 2, 'TIME', props, rotation=90, size=15)
                    props = {'ha': 'left', 'va': 'top',}

                if fig_dir is None:
                    plt.show()
                else:
                    chars = rel.split(' AND ')
                    fig_name = fig_dir + book + \
                        '__' + chars[0] + '__' + chars[1] + '.png'
                    print 'figname = ', fig_name
                    plt.savefig(fig_name)

if __name__ == '__main__':

    wmap, cmap, bmap = cPickle.load(open('data/metadata.pkl', 'rb'))
    rmn_traj = read_csv('models/trajectories.log')  
    rmn_descs = read_descriptors('models/descriptors.log')

    plt.style.use('ggplot')
    color_list = ["peru","dodgerblue","brown","hotpink",
    "aquamarine","springgreen","chartreuse","fuchsia",
    "mediumspringgreen","burlywood","midnightblue","orangered",
    "olive","darkolivegreen","darkmagenta","mediumvioletred",
    "darkslateblue","saddlebrown","darkturquoise","cyan",
    "chocolate","cornflowerblue","blue","red",
    "navy","steelblue","cadetblue","forestgreen",
    "black","darkcyan"]
    color_list += color_list
    plt.rc('axes', prop_cycle=(cycler('color', color_list)))
    viz_csv(rmn_traj, rmn_descs,
        min_length=50, max_viz=True,
        fig_dir='figs/', smallest_shift=1)

