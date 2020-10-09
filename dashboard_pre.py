import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors
from plotly.subplots import make_subplots
import base64
import os

tab_pwd = os.path.join(os.getcwd(),"refined_tab")
#tab_pwd = os.path.join("D:\\insight_project\\code\\insight_project_negative_reviews","refined_tab") 

def plt_tab1(a,b):
    tab1_pwd = os.path.join(tab_pwd, "tab1_1.csv")
    tab1 = pd.read_csv(tab1_pwd , error_bad_lines=False)
    fig = px.scatter_mapbox(tab1, lat="latitude", lon="longitude", hover_name="state", hover_data=["state", "count"], center=go.layout.mapbox.Center(
            lat=a,
            lon=b
        ),
                            color_discrete_sequence=["green"], zoom=11, width = 1300, height=1000, color = "count", size="count")
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    return fig

def plt_tab2():
    #fig = make_subplots(rows=1,cols=2)
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
    fig= go.Figure(data=[go.Table(
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

def plt_tab1_2_2():
    #tab_pwd = os.path.join(os.getcwd(),"refined_tab")
    
    #import datetime

    ab4_pwd = os.path.join(tab_pwd, "tab1_2_2.csv")
    df = pd.read_csv(ab4_pwd)
    df.sort_values("freq", ascending = True)
    #datelist = pd.date_range(datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d'),
    #                         periods=NPERIODS).tolist()
    #df['dates'] = datelist 
    list_1 = ["McDonald's","Starbucks","Chipotle Mexican Grill","Dunkin'","Buffalo Wild Wings"]
    #df = df.set_index(['dates'])
    
    #df.iloc[0] = 0
    #df = df.cumsum()



    # # plotly
    fig = go.Figure()

    # set up ONE trace
    fig.add_trace(go.Bar(y=df[df.business == "McDonald's"]["words"],
                         x=df[df.business == "McDonald's"]["freq"],
                         orientation='h',
                         visible=True
                            ), 
                 )

    updatemenu = []
    buttons = []

    # button with one option for each dataframe
    for col in list_1:
        buttons.append(dict(method='restyle',
                            label=col,
                            visible=True,
                            args=[{'y':[df[df.business == col]["words"]],
                                   'x':[df[df.business == col]["freq"]],
                                   'title':"Negative Reviews for "+col ,
                                   'type':'bar'}, [0]],
                            )
                      )

    # some adjustments to the updatemenus
    updatemenu = []
    your_menu = dict()
    updatemenu.append(your_menu)

    updatemenu[0]['buttons'] = buttons
    updatemenu[0]['direction'] = 'down'
    updatemenu[0]['showactive'] = True
    updatemenu[0]['yanchor'] = "top"
    updatemenu[0]['xanchor'] = "left"
    updatemenu[0]['y'] = 1.1
    updatemenu[0]['x'] = 0
    # add dropdown menus to the figure
    fig.update_layout(showlegend=False, updatemenus=updatemenu, title = "Frequent Words in Top 5 Business's negative reviews:")
    #fig.show()
    return fig

def plt_tab2_1():
    tab2_pwd = os.path.join(tab_pwd, "tab1_2.csv")
    tab2 = pd.read_csv(tab2_pwd , error_bad_lines=False)
    df2 = tab2
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df2.name,
                    x=df2.num_neg_re, orientation='h',
                    name='Negatuve_reviews',
                    marker_color='rgb(55, 83, 109)'
                    ))
    fig.add_trace(go.Bar(y=df2.name,
                    x=df2.num_all_re, orientation='h',
                    name='All Reviews',
                    marker_color='rgb(26, 118, 255)'
                    ))

    fig.update_layout(
        title='Number of reviews received for Businesses',
        xaxis_tickfont_size=12,
        legend=dict(
            x=0.5,
            y=1.0,
            yanchor="top",
            xanchor="left",
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.update_xaxes(tickangle=220)
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
    fig.add_trace(go.Bar(y=df4.name,
                    x=df4.num_neg_re, orientation='h',
                    name='Negatuve_reviews',
                    marker_color='rgb(55, 83, 109)'
                    ))
    fig.add_trace(go.Bar(y=df4.name,
                    x=df4.num_all_re,orientation='h',
                    name='All Reviews',
                    marker_color='rgb(26, 118, 255)'
                    ))

    fig.update_layout(
        title='Number of reviews sent by Users',
        xaxis_tickfont_size=14,
        legend=dict(
            x=0.5,
            y=1.0,
            yanchor="top",
            xanchor="left",
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    #fig.show()
    return fig

def plt_tab1_4_2():
    #tab_pwd = os.path.join(os.getcwd(),"refined_tab")

    #import datetime

    nb4_pwd = os.path.join(tab_pwd, "tab1_4_2.csv")
    df = pd.read_csv(nb4_pwd)

    #datelist = pd.date_range(datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d'),
    #                         periods=NPERIODS).tolist()
    #df['dates'] = datelist 
    list_2 = ["Brad","Jennifer","Stefany","Owen","Diana"]
    #df = df.set_index(['dates'])
    #df.index = pd.to_datetime(df.index)
    #df.iloc[0] = 0
    #df = df.cumsum()



    # # plotly
    fig = go.Figure()

    # set up ONE trace
    fig.add_trace(go.Bar(y=df[df.firstname == "Brad"]["words"],
                         x=df[df.firstname == "Brad"]["freq"],
                         orientation='h',
                             visible=True)
                 )

    updatemenu = []
    buttons = []

    # button with one option for each dataframe
    for col in list_2:
        buttons.append(dict(method='restyle',
                            label=col,
                            visible=True,
                            args=[{'y':[df[df.firstname == col]["words"]],
                                   'x':[df[df.firstname == col]["freq"]],
                                   'type':'bar'}, [0]],
                            )
                      )

    # some adjustments to the updatemenus
    updatemenu = []
    your_menu = dict()
    updatemenu.append(your_menu)

    updatemenu[0]['buttons'] = buttons
    updatemenu[0]['direction'] = 'down'
    updatemenu[0]['showactive'] = True
    updatemenu[0]['yanchor'] = "top"
    updatemenu[0]['xanchor'] = "left"
    updatemenu[0]['y'] = 1.1
    updatemenu[0]['x'] = 0
    
    # add dropdown menus to the figure
    fig.update_layout(showlegend=False, updatemenus=updatemenu, title = "Frequent Words in Top 5 users' negative reviews:")
    #fig.show()
    return fig

def plt_tab5():
    import plotly.express as px
    from skimage import io
    img = io.imread(os.path.join(tab_pwd,"wc.png"))
    fig = px.imshow(img)
    return fig

def plt_tab6_1():
    data6 = pd.read_csv(os.path.join(tab_pwd,"yelp_coef.csv"))
    data6 = data6.sort_values("coef", ascending = True)
    fig = go.Figure()
    list_c = ["neg","pos"]
    drop_name ={"neg":"top 20 words most likely be negative reviews","pos":"top 20 words lease likely be negative reviews"}
    #df = df.set_index(['dates'])
    #df.index = pd.to_datetime(df.index)
    #df.iloc[0] = 0
    #df = df.cumsum()

    # # plotly
    fig = go.Figure()

    # set up ONE trace
    fig.add_trace(go.Bar(x=data6[data6.t == "neg"]["coef"], 
                         y=data6[data6.t == "neg"]["word"],
                         orientation='h',
                         visible=True
                            ), 
                   
                 )

    updatemenu = []
    buttons = []

    # button with one option for each dataframe
    for col in list_c:
        buttons.append(dict(method='restyle',
                            label=drop_name[col],
                            visible=True,
                            args=[{'x':[data6[data6.t== col]["coef"]],
                                   'y':[data6[data6.t == col]["word"]],
                                   'title':"Negative Reviews for "+col ,
                                   'type':'bar'}, [0]],
                            )
                      )

    # some adjustments to the updatemenus
    updatemenu = []
    your_menu = dict()
    updatemenu.append(your_menu)

    updatemenu[0]['buttons'] = buttons
    updatemenu[0]['direction'] = 'down'
    updatemenu[0]['showactive'] = True
    updatemenu[0]['yanchor'] = "top"
    updatemenu[0]['xanchor'] = "left"
    updatemenu[0]['y'] = 1.1
    updatemenu[0]['x'] = 0
    #updatemenu[0]['pad']={"r": 10, "t": 10}
    
    # add dropdown menus to the figure
    fig.update_layout(showlegend=False, updatemenus=updatemenu)
    #fig.show()
    return fig

"""

def plt_tab6_2():
    df = pd.DataFrame([[20, 2],
                       [2, 100]], columns = ["predict_True","predict_False"], index = ["actual_True","actual_False"])

    fig = px.imshow(df,labels=dict(color="Num of Event"))
    return fig
"""
"""
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
"""
    

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://codepen.io/etpinard/pen/QERdjq']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


image_filename = os.path.join(tab_pwd, 'wc.png')
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


image_filename1 = os.path.join(tab_pwd, 'cmroc.png')
encoded_image1 = base64.b64encode(open(image_filename1, 'rb').read())

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Map', children=[
            html.H3(children='Business Map Overview'),
            dcc.Dropdown(id='dropdown', 
                         options=[
                             {'label':'Phoenix', 'value':'Phoenix'},
                             {'label':'Las Vegas','value':'Las Vegas'},
                             {'label':'Toronto','value':'Tronto'}],
                         placeholder='Filter by country...',
                         style={'height': '30px', 'width': '150px'}
                        ),
            dcc.Graph(id='graph_object')
        ]),
        dcc.Tab(label='Businesses', children=[
            html.H3(children='Top companies received most negative reviews'),
            html.Div(children=[
                 dcc.Graph(id="graph1",figure=plt_tab2(),style={'height':'550px','width': '45%', 'display': 'inline-block'}),
                 dcc.Graph(id="graph2",figure=plt_tab2_1(),style={'height':'550px','width': '53%', 'display': 'inline-block'}),
                 dcc.Graph(id="graph2_1",figure=plt_tab1_2_2(),style={'height':'550px','width': '93%', 'display': 'inline-block'})
                                                                  ])
            
        ]),
        dcc.Tab(label='Users', children=[
            html.H3(children='Top complainer sent most negative reviews '),
            html.Div(children=[
                dcc.Graph(figure=plt_tab4(),style={'height':'550px','width': '45%', 'display': 'inline-block'}),
                dcc.Graph(figure=plt_tab4_1(),style={'height':'550px','width': '53%', 'display': 'inline-block'}),
                dcc.Graph(figure=plt_tab1_4_2(),style={'height':'550px','width': '93%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label='Word Cloud', children=[
            html.H3(children='Key words of negative reviews '),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '700px','display': 'flex', 'align':'center'})
        ]),
        dcc.Tab(label='Machine Learning', children=[
            html.H3(children='Sentiment Analysis on negative reviews'),
            html.Div(children=[
                dcc.Graph(figure=plt_tab6_1(), style={'height':'720px','width': '54%','display': 'inline-block',"margin": "auto"}),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image1.decode()), style={'display': 'inline-block',"margin": "auto"})])
                
                
        ]),
    ])
])


@app.callback(dash.dependencies.Output('graph_object', 'figure'),[dash.dependencies.Input('dropdown', 'value')])
def update_graph(value):
    if(value=='Phoenix'):
        return plt_tab1(33.4600,-112.0738)
    elif(value=='Las Vegas'):
        return plt_tab1(36.1169,-115.1832)
    elif(value=='Tronto'):
        return plt_tab1(43.6494,-79.3861)
    else:
        return plt_tab1(33.4600,-112.0738)


if __name__ == '__main__':
    app.run_server(debug=False, host = "0.0.0.0")
