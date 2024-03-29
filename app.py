import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.svm import SVC

import os

app = dash.Dash()
app.layout = html.Div(
            [ 
            html.H1('Hero Help', style={'textAlign': 'center', 'color': 'White', 'background-color': 'rgb(253, 51, 153)'},),
            html.H2('Your Friend in Diagnosing women with cardiovascular Disease', style={'textAlign': 'center', 'color': 'White', 'background-color': 'rgb(253, 51, 153)' }),
            html.Div([ html.A([html.Img(src='https://i.ibb.co/SrV9MhS/logo.png', 
                                        style={
                                              'display': 'block',
                                              'margin-left': 'auto',
                                              'margin-right': 'auto',
                                              'width': '7%',},) 
                          ]),
                ]),
            html.Br(),
            html.Div([
            html.Div([html.P('Personnummer', style={"height": "auto", "margin": "0 auto"}),
                      dcc.Input(id='pn', placeholder = 'YYYYMMDD-XXXX', type="text", value='', ), ]),
            html.Br(),
            html.Div([html.P('Name', style={"height": "auto", "margin": "0 auto"}),
                      dcc.Input(id='name', placeholder = 'First Last', type="text", value='', ), ]),
            html.Br(),
            html.Div([html.P('Doctor ID', style={"height": "auto", "margin": "0 auto"}),
                      dcc.Input(id='did', placeholder = 'XXXXXX', type="text", value='', ), ]),
            html.Br(),
            ],style={'textAlign': 'center','columnCount': 3, 'width' : '30%','height':'80%', 'margin': '0 auto' , 'minWidth' : 1000, 'background-color': 'rgb(60, 185, 200)'}),
            html.Br(),  
            html.Div([
            html.Div([html.P('Age', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='age', placeholder = 'age in years', type="number", value='', ), ]),
            html.Div([html.P('Chest pain type', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='cp', placeholder = '1,2,3,4', type="number", value='', ), ]),
            html.Br(),
            html.Div([html.P('Resting Blood Pressure ', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='tbps', placeholder = 'in mm Hg', type="number", value='', ), ]),
            html.Div([html.P('Serum Cholastoral', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='chol', placeholder = 'serum chol mg/dl', type="number", value='', ), ]),
            html.Div([html.P('Resting ECG result', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='recg', placeholder ='0,1,2', type="number", value='', ), ]),
            html.Div([html.P('Maximum Heart Rate', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='thalach', placeholder ='maximum heart rate achieved', type="number", value='', ), ]),
            html.Div([html.P('Exercise Induced Angina', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='texang', placeholder ='1=yes; 0=no',type="number", value='', ), ]),
            html.Div([html.P('ST Depression', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='oldpeak', placeholder ='1,2,3,4', type="number", value='', ), ]),
        ],style={'textAlign': 'center','columnCount': 3, 'width' : '30%','height':'80%', 'margin': '0 auto' , 'minWidth' : 1000, 'background-color': 'rgb(60, 185, 200)'}),
        html.Br(),
        html.Div(id="result",
                 style={'font-weight': 'bold', 'white-space': 'pre-line','textAlign': 'center','columnCount': 1, 'width' : '30%','height':'90%', 'margin': '0 auto', 'minWidth' : 1000, 'background-color': 'rgb(253, 51, 153)'}),
        html.Br(),
        html.Div([
        html.H3('Cardiovascular Disease Status:  Value 0 = Healthy,   Value 1 = Disease', style={'textAlign': 'center', 'color': 'Black', 'background-color': 'rgb(60, 185, 200)' }),
        ],style={'textAlign': 'center','columnCount': 1, 'width' : '30%','height':'80%', 'margin': '0 auto' , 'minWidth' : 1000, 'background-color': 'rgb(60, 185, 200)'}),
        html.Br(),
        html.Div([
            html.A("HELP", href='https://archive.ics.uci.edu/ml/datasets/heart+disease', target="_blank"),
            ], style={'textAlign': 'center', 'color': 'Red'}),
        html.Br(),
        html.Br(),
        html.Div([
            html.A("toneesha@gmail.com"),
        ], style={'textAlign': 'center', 'color': 'Red'}),
    ],
)


@app.callback(
     Output('result', 'children'),
     [Input('age', 'value'),
     Input('cp', 'value'),
     Input('tbps', 'value'),
     Input('chol', 'value'),
     Input('recg', 'value'),
     Input('thalach', 'value'),
     Input('texang', 'value'),
     Input('oldpeak', 'value'),
     ]
)
def update_result(age, cp, tbps, chol, recg, thalach, texang, oldpeak):

    dataFile1 = ["processed.hungarian.data", "processed.cleveland.data", "processed.switzerland.data", "processed.va.data"]
    i=  0 #----> change file
    File = (dataFile1[i])
    data1 = pd.read_csv(File, header=None)
    data1.set_axis(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'], axis=1,inplace=True)
    data1 = data1.drop('ca', axis=1)
    data1 = data1.drop("slope", axis=1)
    data1 = data1.drop("thal", axis=1)
    data1 = data1.drop("fbs", axis=1)
    data1 = data1[data1!="?"].dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)
    def replace_num(x):
        if x == "?":
            return 0
        elif x == "1" or x =="1.0" or x == "2" or x == 2 or x =="2.0" or x == "3" or x =="3.0" or x == 3 or x == 4 or x == "4" or x =="4.0":
            return "1"
        else:
            return x
    data1["num"] = data1["num"].apply(replace_num)

    pd.set_option("display.float", "{:.2f}".format)
    data=data1
    data[["sex","cp","restecg","exang","num"]] = data1[["sex","cp","restecg","exang","num"]].astype(int)
    data[["age","trestbps","chol","thalach","oldpeak"]] = data1[["age","trestbps","chol","thalach","oldpeak"]].astype(float)
    dataSet1 = data

    dataFile1 = ["processed.hungarian.data", "processed.cleveland.data", "processed.switzerland.data", "processed.va.data"]
    i=  1
    File = (dataFile1[i])

    data1 = pd.read_csv(File, header=None)
    data1.set_axis(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'], axis=1,inplace=True)
    data1 = data1.drop('ca', axis=1)
    data1 = data1.drop("slope", axis=1)
    data1 = data1.drop("thal", axis=1)
    data1 = data1.drop("fbs", axis=1)
    data1 = data1[data1!="?"].dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)
    def replace_num(x):
        if x == "?":
            return 0
        elif x == "1" or x =="1.0" or x == "2" or x == 2 or x =="2.0" or x == "3" or x =="3.0" or x == 3 or x == 4 or x == "4" or x =="4.0":
            return "1"
        else:
            return x
    data1["num"] = data1["num"].apply(replace_num)

    pd.set_option("display.float", "{:.2f}".format)
    data=data1
    data[["sex","cp","restecg","exang","num"]] = data1[["sex","cp","restecg","exang","num"]].astype(int)
    data[["age","trestbps","chol","thalach","oldpeak"]] = data1[["age","trestbps","chol","thalach","oldpeak"]].astype(float)
    dataSet2 = data

    train = dataSet2
    test = dataSet1

    X_train_F = train[train["sex"]==0]
    X_train_F = X_train_F.drop('num', axis=1)
    X_train_F = X_train_F.drop('sex', axis=1)
    y_train_F = train[train["sex"]==0]['num']

    X_test_F = test[test["sex"]==0]
    X_test_F1 = X_test_F.drop('num', axis=1)
    X_test_F = X_test_F1.drop('sex', axis=1)

    y_test_F = test[test["sex"]==0]['num']

    svm_linear = SVC(kernel='linear', gamma=0.1, C=1.0)
    svm_linear.fit(X_train_F, y_train_F)

    X_test_F = [[age,cp,tbps,chol,recg,thalach,texang,oldpeak]]
    y_pred = svm_linear.predict(X_test_F)    
    
    return "\n Possibilty of Cardiovascular disease = %s \n -" % (y_pred[0])
if __name__ == '__main__':
            app.run_server(debug=True, host='0.0.0.0', port=os.environ.get('PORT', '5000'))
