import dash
import dash_html_components as html
import dash_core_components as dcc

import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors
import os

tab_pwd = os.path.join(os.getcwd(),"refined_tab")
#tab_pwd = os.path.join("D:\\insight_project\\tabs","refined_tab")
def plt_tab1():
    tab1_pwd = os.path.join(tab_pwd, "tab1_1.csv")
    tab1 = pd.read_csv(tab1_pwd , error_bad_lines=False)
    fig = px.scatter_mapbox(tab1, lat="latitude", lon="longitude", hover_name="state", hover_data=["state", "count"], center=go.layout.mapbox.Center(
            lat=33.4600,
            lon=-112.0738
        ),
                            color_discrete_sequence=["green"], zoom=11, width = 1300, height=1000, color = "count", size="count")
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    return fig

def plt_tab2():
    tab2_pwd = os.path.join(tab_pwd, "tab1_2.csv")
    tab2 = pd.read_csv(tab2_pwd , error_bad_lines=False)
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

def plt_tab2_1():
    tab2_pwd = os.path.join(tab_pwd, "tab1_2.csv")
    tab2 = pd.read_csv(tab2_pwd , error_bad_lines=False)
    df2 = tab2
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df2.name,
                    y=df2.num_neg_re,
                    name='Negatuve_reviews',
                    marker_color='rgb(55, 83, 109)'
                    ))
    fig.add_trace(go.Bar(x=df2.name,
                    y=df2.num_all_re,
                    name='All Reviews',
                    marker_color='rgb(26, 118, 255)'
                    ))

    fig.update_layout(
        title='Number of reviews received for Businesses',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Freq',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    #fig.show()
    return fig

def plt_tab3():
    tab3_pwd = os.path.join(tab_pwd, "tab1_3.csv")
    df_tab3 = pd.read_csv(tab3_pwd , error_bad_lines=False)
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
    tab4_pwd = os.path.join(tab_pwd, "tab1_4.csv")
    df4 = pd.read_csv(tab4_pwd , error_bad_lines=False)
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

def plt_tab4_1():
    tab4_pwd = os.path.join(tab_pwd, "tab1_4.csv")
    df4 = pd.read_csv(tab4_pwd , error_bad_lines=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df4.name,
                    y=df4.num_neg_re,
                    name='Negatuve_reviews',
                    marker_color='rgb(55, 83, 109)'
                    ))
    fig.add_trace(go.Bar(x=df4.name,
                    y=df4.num_all_re,
                    name='All Reviews',
                    marker_color='rgb(26, 118, 255)'
                    ))

    fig.update_layout(
        title='Number of reviews sent by Users',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Freq',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    #fig.show()
    return fig

def plt_tab5():
    import plotly.express as px
    from skimage import io
    img = io.imread(os.path.join(tab_pwd,"wc.png"))
    fig = px.imshow(img)
    return fig

def plt_tab6_1():
    data6 = pd.read_csv(os.path.join(tab_pwd, "feature_importance.csv"))
    
    #data_canada["lifeExp"] -= 75

    fig = go.Figure()
    fig.add_trace(go.Bar(x=data6.word, y=-data6.coef,
                    base=data6.coef,
                    marker_color='crimson',
                    name='features'))
    return fig

def plt_tab6_2():
    df = pd.DataFrame([[20, 2],
                       [2, 100]], columns = ["predict_True","predict_False"], index = ["actual_True","actual_False"])

    fig = px.imshow(df,labels=dict(color="Num of Event"))
    return fig


def plt_tab6_3():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, random_state=0)

    model = LogisticRegression()
    model.fit(X, y)
    y_score = model.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
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
            ),
            dcc.Graph(
                figure=plt_tab2_1()
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
            ),
            dcc.Graph(
                figure=plt_tab4_1()
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
                figure=plt_tab6_1()
            ),
            dcc.Graph(
                figure=plt_tab6_2()
            ),
            dcc.Graph(
                figure=plt_tab6_2()
            )

        ]),
    ])
])


#if __name__ == '__main__':
#    app.run_server(debug=True)