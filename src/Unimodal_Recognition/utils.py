import numpy as np
def plot_confusion_by_heat_map(matrix, xaxis, yaxis, fig_name):
    """
    drawing confusion matrix with heatmap
    :param matrix: confusion matrix
    :param xaxis: x axis label
    :param yaxis: y axis label
    :return:
    """
    from pyecharts import options as opts
    from pyecharts.charts import HeatMap
    (row, col) = matrix.shape
    data = []
    for i in range(row):
        for j in range(col):
            data.append([i, j, np.float32(matrix[i][j]).item()])
    heat_map = (
        HeatMap()
        .add_xaxis(xaxis)
        .add_yaxis('f1_score', yaxis, data)
        .set_global_opts(
            title_opts=opts.TitleOpts(title=fig_name),
            visualmap_opts=opts.VisualMapOpts(range_color=['#121122', 'rgba(3,4,5,0.4)', 'red']),
        )
    )
    heat_map.render(path='../chart/' + fig_name + '.html')