#!/usr/bin/env python
# coding: utf-8

# In[29]:


import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors


def plt_tab4():
    df4 = pd.read_csv("D:\\insight_project\\tabs\\refined_tab\\tab1_4.csv", error_bad_lines=False)
    colors_a = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 2, colortype='rgb')
    colors_b = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 578, colortype='rgb')
    colors_c = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 4332, colortype='rgb')
    colors_d = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 100, colortype='rgb')
    a = df4.name
    a_1 = df4.fans
    b = df4.num_neg_re
    c = df4.num_all_re
    d =  [round(x, 2) for x in df4.percent]
    d_v = [round(x*1) for x in df4.percent]
    fig = go.Figure(data=[go.Table(
      header=dict(
        values=['<b>Fisrt Name</b>','<b>Fans</b>', '<b>Negative Reviews</b>', '<b>Total Reviews</b>','<b>Percentage</b>'],
        line_color='white', fill_color='pink',
        align='center',font=dict(color='black', size=12)
      ),
      cells=dict(
        values=[a,a_1, b, c, d],
        line_color=[np.array(colors_a)[0],np.array(colors_a)[0],np.array(colors_b)[b],np.array(colors_c)[c], np.array(colors_d)[d_v]],
        fill_color=[np.array(colors_a)[0],np.array(colors_a)[0],np.array(colors_b)[b],np.array(colors_c)[c], np.array(colors_d)[d_v]],
        align='center', font=dict(color='black', size=11)
        ))
    ])
    #fig.show()
    return fig





