import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import plotly.express as px

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



app = dash.Dash()

app.layout = html.Div(
            [
            html.Br(),
            html.Br(),   
            html.H1('Hero Helper Prediction Engine', style={'textAlign': 'center', 'color': 'Blue', 'background-color': 'rgb(176, 168, 215)'}),
#        html.H1('Prediction Engine', style={'textAlign': 'center', 'color': 'Blue'}),
            html.Br(),
            html.Br(),  
            html.Div([
            html.Div([html.P('Age: age in years', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='age', placeholder = 'age in years', type="number", value='', ), ]),
#            html.Div([html.P('sex:(1 = male; 0 = female)', style={"height": "auto", "margin-bottom": "auto"}),
#                      dcc.Input(id='sex', placeholder = '1 = male; 0 = female', type="number", value='', ), ]),
            html.Div([html.P('cp: chest pain type 1,2,3,4', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='cp', placeholder = '1,2,3,4', type="number", value='', ), ]),
            html.Div([html.P('trestbps: resting blood pressure (in mm Hg on admission to the hospital)', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='tbps', placeholder = 'in mm Hg', type="number", value='', ), ]),
            html.Div([html.P('chol: serum cholestoral in mg/dl', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='chol', placeholder = 'serum chol mg/dl', type="number", value='', ), ]),
#            html.Div([html.P('fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)', style={"height": "auto", "margin-bottom": "auto"}),
#                      dcc.Input(id='fbs', placeholder = '> 120 mg/dl 1=true; 0=false', type="number", value='', ), ]),
            html.Br(),
            html.Br(),
            html.Div([html.P('restecg: resting electrocardiographic resultsres -0,1,2', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='recg', placeholder ='0,1,2', type="number", value='', ), ]),
            html.Div([html.P('thalach: maximum heart rate achieved', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='thalach', placeholder ='maximum heart rate achieved', type="number", value='', ), ]),
            html.Div([html.P('exang: exercise induced angina (1 = yes; 0 = no)', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='texang', placeholder ='exercise induced angina  1=yes; 0=no',type="number", value='', ), ]),
            html.Div([html.P('oldpeak = ST depression induced by exercise relative to rest', style={"height": "auto", "margin-bottom": "auto"}),
                      dcc.Input(id='oldpeak', placeholder ='ST depression induced',type="number", value='', ), ]),
#            html.Div([html.P('slope: the slope of the peak exercise ST segment - 1,2,3', style={"height": "auto", "margin-bottom": "auto"}),
#                      dcc.Input(id='slope', placeholder ='1,2,3',type="number", value='', ), ]),
#            html.Div([html.P('ca: number of major vessels (0-3) colored by flourosopy', style={"height": "auto", "margin-bottom": "auto"}),
#                      dcc.Input(id='ca', placeholder ='num of major vessels 0-3', type="number", value='', ), ]),
#            html.Div([html.P('thal: 3 = normal; 6 = fixed defect; 7 = reversable defect', style={"thal": "auto", "margin-bottom": "auto"}),
#                      dcc.Input(id='thal', placeholder ='3 = normal; 6 = fixed defect; 7 = reversable defect', type="number", value='', ), ]),
            html.Br(),
#        ], style=dict(display='flex', flexWrap='wrap', width=400,)),
        ],style={'textAlign': 'center','columnCount': 2, 'width' : '30%','height':'80%', 'marginLeft': 160 , 'minWidth' : 1000, 'background-color': 'rgb(176, 168, 215)'}),
#        ],style={'columnCount': 2,'textAlign': 'center','display':'table-cell','padding':5, 'verticalAlign':'middle'}),


#        html.Div([html.Button('Submit', style={"height": "auto", "margin-bottom": "auto"},
#                              id='button'), ]),
        #html.Button('Submit', id='button'),
        html.Br(),
        html.Div(id="result",
                 style={'font-weight': 'bold','white-space': 'pre-line','textAlign': 'center','columnCount': 1, 'width' : '30%','height':'80%', 'marginLeft': 160 , 'minWidth' : 1000, 'background-color': 'rgb(200, 200, 152)'}),
        html.Div([
            html.A("HELP", href='https://archive.ics.uci.edu/ml/datasets/heart+disease', target="_blank"),
            ]),
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
def update_result(age,cp, tbps, chol, recg, thalach, texang, oldpeak):

    """"
    df = pd.read_csv('C:\\Users\\Senithma\\Desktop\\Python\\processed.cleveland.data.csv')
    dfd = df[(df["ca"]!="?")&(df["thal"]!="?")]
#   dfd["num"].replace({2:1,3:1,4:1}, inplace=True)

    X = dfd.drop('num',axis=1)
    y = dfd['num']

# implementing train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)
    rfc = RandomForestClassifier(max_depth=2,random_state=0)
    rfc.fit(X_train,y_train)
# predictions
    
    X_test = [[age,sex,cp,tbps,chol,fbs,recg,thalach,texang,oldpeak,slope,ca,thal]]
    y_pred = rfc.predict(X_test)
    """

    dataFile1 = ["processed.hungarian.data", "processed.cleveland.data", "processed.switzerland.data", "processed.va.data"]
    i=  0 #----> change file
    File = (dataFile1[i])

    data1 = pd.read_csv(File, header=None)
    data1.set_axis(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'], axis=1,inplace=True)
#delete some irrelevant features
    data1 = data1.drop('ca', axis=1)
    data1 = data1.drop("slope", axis=1)
    data1 = data1.drop("thal", axis=1)
    data1 = data1.drop("fbs", axis=1)
#delete missing values
    data1 = data1[data1!="?"].dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)

# FUNCTION (replace ? & 1, 2, 3,4) (constant 0,1)
    def replace_num(x):
        if x == "?":
            return 0
        elif x == "1" or x =="1.0" or x == "2" or x == 2 or x =="2.0" or x == "3" or x =="3.0" or x == 3 or x == 4 or x == "4" or x =="4.0":
            return "1"
        else:
            return x
#-------------------------------------------------
    data1["num"] = data1["num"].apply(replace_num)

##3. Unify Datatype
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
#delete some irrelevant features
    data1 = data1.drop('ca', axis=1)
    data1 = data1.drop("slope", axis=1)
    data1 = data1.drop("thal", axis=1)
    data1 = data1.drop("fbs", axis=1)
#delete missing values
    data1 = data1[data1!="?"].dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)

# FUNCTION (replace ? & 1, 2, 3,4) (constant 0,1)
    def replace_num(x):
        if x == "?":
            return 0
        elif x == "1" or x =="1.0" or x == "2" or x == 2 or x =="2.0" or x == "3" or x =="3.0" or x == 3 or x == 4 or x == "4" or x =="4.0":
            return "1"
        else:
            return x
#-------------------------------------------------
    data1["num"] = data1["num"].apply(replace_num)

##3. Unify Datatype
    pd.set_option("display.float", "{:.2f}".format)
    data=data1
    data[["sex","cp","restecg","exang","num"]] = data1[["sex","cp","restecg","exang","num"]].astype(int)
    data[["age","trestbps","chol","thalach","oldpeak"]] = data1[["age","trestbps","chol","thalach","oldpeak"]].astype(float)
    dataSet2 = data

#train = dataSet1.append(dataSet2)
    train = dataSet2
#test = dataSet3
    test = dataSet1

# Femenine - train
    X_train_F = train[train["sex"]==0]
    X_train_F = X_train_F.drop('num', axis=1)
    X_train_F = X_train_F.drop('sex', axis=1)

    y_train_F = train[train["sex"]==0]['num']

# Femenine - test
    X_test_F = test[test["sex"]==0]
    X_test_F1 = X_test_F.drop('num', axis=1)
    X_test_F = X_test_F1.drop('sex', axis=1)

    y_test_F = test[test["sex"]==0]['num']


    random_forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=12)
    random_forest.fit(X_train_F, y_train_F)

#    X_test_F = [[56,1,125,210,2,165,2,2]]
    X_test_F = [[age,cp,tbps,chol,recg,thalach,texang,oldpeak]]
    y_pred = random_forest.predict(X_test_F)

    return "Input Dataset=%s, model-RandomForest, \n diagnosis of heart disease (angiographic disease status)=%s" % (X_test_F[0], y_pred[0]) 


app.run_server(debug=False)
