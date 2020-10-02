#!/usr/bin/env python
# coding: utf-8

# In[10]:


import plotly
import pandas as pd
import numpy as np
import plotly.express as px


def plt_tab1():
    tab1 = pd.read_csv("D:\\insight_project\\tabs\\refined_tab\\tab1_1.csv", error_bad_lines=False)
    fig = px.scatter_mapbox(tab1, lat="latitude", lon="longitude", hover_name="state", hover_data=["state", "count"],
                            color_discrete_sequence=["green"], zoom=3, height=300, color = "count", size="count")
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    return fig

#fig1 = plt_tab1()





