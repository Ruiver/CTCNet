"""
author: ouyangtianxiong
date:2019/12/26
des:visualizing raw feature data of seed
"""
import sys
sys.path.append('../')
import numpy as np
from data_set.seed_iv import SEED_IV
from pyecharts.charts import Line, Page
import pyecharts.options as opts
import os


def numpy_vector_to_python(data):
    """
    numpy vector to python data struct
    :param data:
    :return:
    """
    data = [np.float32(e).item() for e in data]
    return data

def plot(session, individual):
    """
    visualizing data of individual in session
    :param session:
    :param individual:
    :return:
    """
    eeg = SEED_IV(session=session, individual=individual, modal='concat', shuffle=False, balance=False,
                  normalization=1)
    X, Y = eeg.get_X_Y()
    eeg = X[:, :310]
    eye = X[:, 310:]
    eeg = eeg.reshape(-1, 62, 5)
    b, n, d = eeg.shape
    page = Page()
    # 眼动特征可视化
    # 1-12瞳孔直径
    ling_pupil = (
        Line()
        .add_xaxis([i for i in range(b)])
        .add_yaxis('pupil diameter 1', numpy_vector_to_python(eye[:,0]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter 2', numpy_vector_to_python(eye[:,1]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter 3', numpy_vector_to_python(eye[:,2]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter 4', numpy_vector_to_python(eye[:,3]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter mean 1', numpy_vector_to_python(eye[:,4]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter mean 2', numpy_vector_to_python(eye[:,5]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter mean 3', numpy_vector_to_python(eye[:,6]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter mean 4', numpy_vector_to_python(eye[:,7]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter std 1', numpy_vector_to_python(eye[:,8]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter std 2', numpy_vector_to_python(eye[:,9]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter std 3', numpy_vector_to_python(eye[:,10]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('pupil diameter std 4', numpy_vector_to_python(eye[:,11]), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title='pupil diameter feature'), legend_opts=opts.LegendOpts())
    )
    page.add(ling_pupil)
    # dispersion
    line_dispersion = (
        Line()
        .add_xaxis([i for i in range(b)])
        .add_yaxis('dispersion 1', numpy_vector_to_python(eye[:,12]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('dispersion 2', numpy_vector_to_python(eye[:,13]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('dispersion 3', numpy_vector_to_python(eye[:,14]), label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('dispersion 4', numpy_vector_to_python(eye[:,15]), label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title='dispersion'), legend_opts=opts.LegendOpts())
    )
    page.add(line_dispersion)

    #fixation duration
    line_fixation = (
        Line()
            .add_xaxis([i for i in range(b)])
            .add_yaxis('fixation 1', numpy_vector_to_python(eye[:,16]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('fixation 2', numpy_vector_to_python(eye[:,17]), label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title='fixation'), legend_opts=opts.LegendOpts())
    )
    page.add(line_fixation)

    # saccade
    line_saccade = (
        Line()
            .add_xaxis([i for i in range(b)])
            .add_yaxis('saccade 1', numpy_vector_to_python(eye[:,18]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('saccade 2', numpy_vector_to_python(eye[:,19]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('saccade 3', numpy_vector_to_python(eye[:,20]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('saccade 4', numpy_vector_to_python(eye[:,21]), label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title='saccade'), legend_opts=opts.LegendOpts())
    )
    page.add(line_saccade)

    # event statistics
    line_event = (
        Line()
            .add_xaxis([i for i in range(b)])
            .add_yaxis('event 1', numpy_vector_to_python(eye[:,22]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 2', numpy_vector_to_python(eye[:,23]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 3', numpy_vector_to_python(eye[:,24]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 4', numpy_vector_to_python(eye[:,25]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 5', numpy_vector_to_python(eye[:,26]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 6', numpy_vector_to_python(eye[:,27]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 7', numpy_vector_to_python(eye[:,28]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 8', numpy_vector_to_python(eye[:,29]), label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('event 9', numpy_vector_to_python(eye[:,30]), label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title='event'), legend_opts=opts.LegendOpts())
    )
    page.add(line_event)
    # 62个电极的信息，每个电极的信息，画出五种波
    for i in range(62):
        line = (
            Line()
                .add_xaxis([i for i in range(b)])
                .add_yaxis('delta wave', numpy_vector_to_python(eeg[:, i, 0]), color='red', label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis('theta wave', numpy_vector_to_python(eeg[:, i, 1]), color='yellow', label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis('alpha wave', numpy_vector_to_python(eeg[:, i, 2]), color='green', label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis('beta wave', numpy_vector_to_python(eeg[:, i, 3]), color='blue', label_opts=opts.LabelOpts(is_show=False))
                .add_yaxis('gamma wave', numpy_vector_to_python(eeg[:, i, 4]), color='black', label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(title_opts=opts.TitleOpts(title='electrode %d' % i), legend_opts=opts.LegendOpts())
        )
        page.add(line)
    save_path = '../../chart/Raw_data_visualization/%d/%d' % (session, individual)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    page.render(path=os.path.join(save_path, '%d_%d_norm_feature.html' % (session, individual)))

def main():
    for i in range(1,4):
        for j in range(1,16):
            plot(i,j)

if __name__ == '__main__':
    main()