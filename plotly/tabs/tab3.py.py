#!/usr/bin/env python
# coding: utf-8

# In[27]:


import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors


def plt_tab3():
    df_tab3 = pd.read_csv("D:\\insight_project\\tabs\\refined_tab\\tab1_3.csv", error_bad_lines=False)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_tab3.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_tab3.Word, df_tab3.Frequecy],
                   fill_color='lavender',
                   align='left'))
    ])

    #fig.show() 
    return fig

#fig3 = plt_tab1()






