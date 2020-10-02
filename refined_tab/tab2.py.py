#!/usr/bin/env python
# coding: utf-8

# In[20]:


import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors


def plt_tab2():
    tab2 = pd.read_csv("D:\\insight_project\\tabs\\refined_tab\\tab1_2.csv", error_bad_lines=False)
    df2 = tab2
    #np.random.seed(1)
    colors_a = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 2, colortype='rgb')
    colors_b = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 31407, colortype='rgb')
    colors_c = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 21000, colortype='rgb')
    colors_d = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 100, colortype='rgb')
    a = df2.name
    b = df2.num_all_re
    c = df2.num_neg_re
    d = [round(x,2) for x in df2.percent]
    d_v = [round(x*1) for x in df2.percent]
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Business Name</b>', '<b>Total Reviews</b>', '<b>Negative Reviews</b>','<b>Percentage</b>'],
        line_color='white', fill_color='pink',
        align='center',font=dict(color='black', size=12)
      ),
      cells=dict(
        values=[a, b, c, d_v],
        line_color=[np.array(colors_a)[0],np.array(colors_b)[b],np.array(colors_c)[c], np.array(colors_d)[d_v]],
        fill_color=[np.array(colors_a)[0],np.array(colors_b)[b],np.array(colors_c)[c], np.array(colors_d)[d_v]],
        align='center', font=dict(color='black', size=11)
        ))
    ])

    #fig.show()
    return fig

#fig2 = plt_tab1(tab2)





