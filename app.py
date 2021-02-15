import os
import time
import pandas as pd
import re
import pickle
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
from transformers import BartTokenizer, BartForConditionalGeneration
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch

import joblib 
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load Model
pretrained = "sshleifer/distilbart-xsum-12-6"
model_bart = BartForConditionalGeneration.from_pretrained(pretrained)
tokenizer = BartTokenizer.from_pretrained(pretrained)

# Switch to cuda, eval mode, and FP16 for faster inference
if device == "cuda":
    model_bart = model_bart.half()
model_bart.to(device)
model_bart.eval();


model = pickle.load(open('model.pkl', 'rb'))
tfidf = joblib.load('tfidf.joblib')

app =  dash.Dash(external_stylesheets = [dbc.themes.CERULEAN])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink("Home", href="/main"))
    ],
    brand="U.S. Census Bureau E-mail NLP Dash",
    brand_href="#",
    color="primary",
    dark=True,
)


row_1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Container([html.H3("Subject Line")])),
                dbc.Col(
                    dbc.Container([html.H3("Email Body")])),
            ]
        ),
    ]
)

form = dbc.InputGroup([
            dbc.Input(
                id='input-box',
                placeholder='Enter your subject line',
                type='text',
                value=''), 
            dbc.InputGroupAddon(
                dbc.Button('Submit', id='button'),
                addon_type="append"),
        ])

cards = dbc.CardDeck(
    [
          dbc.Card([
            dbc.CardBody(
                [
                 html.H5("Open Rate", className="card-title"),
                 html.P(
                        "Figure out the predicted open rate of your subject line",
                        className="card-text",
                    ),
                ],
                ),
            dbc.CardFooter(id='open_output')    
          ]),
          dbc.Card([
            dbc.CardBody(
                [
                 html.H5("Sentiment", className="card-title"),
                 html.P(
                        "Figure out the predicted sentiment of your subject line",
                        className="card-text",
                    ),
                 ],
                 ),
            dbc.CardFooter(id='sentiment_output')
          ]), 
          dbc.Card([
            dbc.CardBody(
                [
                 html.H5("Count", className="card-title"),
                 html.P(
                        "Figure out the word and character count of your subject line",
                        className="card-text",
                    ),
                 ],
                 ),
            dbc.CardFooter(id='count_output')
          ]),
    ]
)

text_area = dbc.InputGroup(
            [
                dbc.Textarea(placeholder = "Enter your email content here..."),
                dbc.InputGroupAddon(dbc.Button('Submit', id='button-2'), addon_type="append"),
               
            ],
            className="mb-3",
        )


controls = dbc.Container(
    [
        dbc.FormGroup(
            [
                dbc.Label("Output Length (Max)"),
                dcc.Slider(
                    id="max-length",
                    min=10,
                    max=50,
                    value=30,
                    marks={i: str(i) for i in range(10, 51, 10)},
                ),
              dbc.Label("Output Length (Min)"),
                dcc.Slider(
                    id="min-length",
                    min=10,
                    max=50,
                    value=10,
                    marks={i: str(i) for i in range(10, 51, 10)},
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Branch Size"),
                dcc.Slider(
                    id="num-beams",
                    min=2,
                    max=8,
                    value=4,
                    marks={i: str(i) for i in [2, 4, 6, 8]},
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Spinner(
                    [
                        dbc.Button("Summarize", id="button-run"),
                        html.Div(id="time-taken"),
                    ]
                )
            ]
        ),
    ]
)

sum_content = dbc.Container([
                html.H4("Summarized Content"),
                html.Br(),
                dbc.Textarea(id="summarized-content")
            ])


orig_content = dbc.Container([
                    html.H4("Original Text (Paste here)"),
                    html.Br(),
                    dbc.Textarea(id="original-text")
                    ])

app.layout = dbc.Container([
        navbar,
        row_1,
        dbc.Row([
            dbc.Col(
                dbc.Container([
                    dbc.Row(form),
                    html.Br(),
                    dbc.Row(cards)
                    ])
            ),
            dbc.Col(
                dbc.Container([
                    dbc.Row(orig_content),
                    html.Br(),
                    html.Center(
                      dbc.Row(controls)),
                    dbc.Row(sum_content),
                    html.Br()]),
                    ),
                    ])
    ], fluid = True)

@app.callback(
    dash.dependencies.Output('open_output', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')],
    )

def update_output(n_clicks, input_box):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    elif input_box == "":
        return "Please enter a subject line and press submit"
    else:
        X = [input_box]  
        tfidf_vect = tfidf
        text = tfidf_vect.transform(X, copy = True).toarray()
        result = str(model.predict(text))
        result = result.strip("[]()'")
        result = f"Predicted open rate between: {result}"
        return result

@app.callback(
    dash.dependencies.Output('sentiment_output', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')],
    )

def update_sentiment(n_clicks, value):
    if n_clicks > 0:
        sentence = value 
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(sentence)
        if vs['compound'] >=0.05:
            return "positive sentiment"
        elif vs['compound'] >-0.05 and vs['compound'] <0.05:
            return "neutral sentiment"
        elif vs['compound'] <=-0.05:
            return "negative sentiment"


@app.callback(
    dash.dependencies.Output('count_output', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')],
    )

def update_count(n_clicks, value):
    if n_clicks > 0:
        sentence = value 
        count = len(sentence.split()) 
        length = len(sentence)
        count_final = f"Word count: {count} | Character Count: + {length}"
        return count_final


@app.callback(
    [Output("summarized-content", "value"), Output("time-taken", "children")],
    [
        Input("button-run", "n_clicks"),
        Input("max-length", "value"),
        Input("min-length", "value"),
        Input("num-beams", "value"),
    ],
    [State("original-text", "value")],
)

def summarize(n_clicks, max_length, min_length, num_beams, original_text):
    if original_text is None or original_text == "":
        return None

    t0 = time.time()

    inputs = tokenizer.batch_encode_plus(
        [original_text], max_length=1024, return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Generate Summary
    summary_ids = model_bart.generate(
        inputs["input_ids"],
        num_beams=num_beams,
        min_length=min_length,
        max_length=max_length,
        early_stopping=True,
    )
    out = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for g in summary_ids
    ]

    t1 = time.time()
    time_taken = f"Summarized on {device} in {t1-t0:.2f}s"

    return out[0], time_taken

app.run_server(mode='external', debug = False)


if __name__ == '__main__':
    app.run_server()