"""
author: ouyang tianxiong
date: 2019/12/17
des: implement utilities tools evaluation performance of model
"""
import sys
sys.path.append('../')
from pyecharts.charts import Line,Page
import pyecharts.options as opts
import pandas as pd
import numpy as np
import os
import time

def plot_acc_loss_curve(acc_loss_dict,model_name,fig_name):
    """
    plot loss and acc curve for model training
    :param acc_loss_dict:
    :param model_name: model been evaluated
    :param fig_name: experiment name
    :return: render a html web page contain visualization curve
    """
    for e in ['train_loss','train_acc',  'test_loss','test_acc']:
        assert e in acc_loss_dict.keys(), 'data dict mistake'
    train_len = len(acc_loss_dict['train_loss'])
    t = time.localtime(time.time())

    page = Page()
    loss_line = (
        Line()
        .add_xaxis([i for i in range(train_len)])
        .add_yaxis('训练集loss', acc_loss_dict['train_loss'],label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis('测试集loss', acc_loss_dict['test_loss'],label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title='loss curve'))
    )
    acc_line = (
        Line()
            .add_xaxis([i for i in range(train_len)])
            .add_yaxis('训练集acc', acc_loss_dict['train_acc'],label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis('测试集acc', acc_loss_dict['test_acc'],label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title='acc curve'),legend_opts=opts.LegendOpts())
    )
    page.add(loss_line)
    page.add(acc_line)
    chart_path = '../../chart/%s/%s_%s'% (model_name,t.tm_mon,t.tm_mday)
    if not os.path.exists(chart_path):
        os.makedirs(chart_path)
    page.render(path=os.path.join(chart_path, fig_name+'.html'))


def write_result(model_name):
    length = len(model_name)+1
    res = np.zeros(shape=(2, 3, 15))
    for i, mode in enumerate(["subject_independent", "subject_dependent"]):
        for session in range(1,4):
            for idx in range(1, 16):
                if mode == "subject_dependent":
                    result_path = "../../saved_models/%s/session_%d/subject_%d" % (model_name,session,idx)
                else:
                    result_path = "../../saved_models/%s/session_%d/subject_%d_as_testset" % (model_name,session,idx)
                files = os.listdir(result_path)
                best = max(files)
                acc_str = "0." + best[length:]
                acc = np.round(float(acc_str), decimals=4)
                res[i][session-1][idx-1] = acc
    import pandas as pd
    res1 = res[0].squeeze()
    print(res1)
    mean1 = np.mean(res1, axis=1).reshape(3,1)
    std1 = np.std(res1, axis=1).reshape(3,1)
    res1 = np.hstack([res1, mean1, std1])

    res2 = res[1].squeeze()
    mean2 = np.mean(res2, axis=1).reshape(3,1)
    std2 = np.std(res2, axis=1).reshape(3,1)
    res2 = np.hstack([res2, mean2, std2])
    pd.DataFrame(res1).to_csv("subject_independent.csv",header=False,index=False,mode='w',encoding='utf-8')
    pd.DataFrame(res2).to_csv("subject_dependent.csv",header=False,index=False,mode='w',encoding='utf-8')

def main():
    write_result('Hierarchical_ATTN')

if __name__ == '__main__':
    main()
