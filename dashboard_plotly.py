import dash
import dash_html_components as html
import dash_core_components as dcc

import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors

def plt_tab1():
    tab1 = pd.read_csv("D:\\insight_project\\tabs\\refined_tab\\tab1_1.csv", error_bad_lines=False)
    fig = px.scatter_mapbox(tab1, lat="latitude", lon="longitude", hover_name="state", hover_data=["state", "count"],
                            color_discrete_sequence=["green"], zoom=3, height=300, color = "count", size="count")
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    return fig

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

def plt_tab5():
    import plotly.express as px
    from skimage import io
    img = io.imread('D:\\insight_project\\tabs\\refined_tab\\wc.png')
    fig = px.imshow(img)
    return fig


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Map', children=[
            html.H3(children='Business Map Overview'),
            dcc.Graph(
                figure=plt_tab1()
            )
        ]),
        dcc.Tab(label='Company', children=[
            html.H3(children='Companies received most negative reviews'),
            dcc.Graph(
                figure=plt_tab2()
            )
        ]),
        dcc.Tab(label='Words', children=[
            html.H3(children='Word Counts'),
            dcc.Graph(
                figure=plt_tab3()
            )
        ]),
        dcc.Tab(label='Complaiter', children=[
            html.H3(children='Complainer sent most negative reviews '),
            dcc.Graph(
                figure=plt_tab4()
            )
        ]),
        dcc.Tab(label='Word Cloud', children=[
            html.H3(children='Key words of negative reviews '),
            dcc.Graph(
                figure=plt_tab5()
            )
        ]),
        dcc.Tab(label='Machine Learning', children=[
            html.H3(children='Sentiment Analysis on negative reviews'),
            dcc.Graph(
                figure=plt_tab4()
            )
        ]),
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)