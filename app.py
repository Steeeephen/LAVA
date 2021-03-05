import flask
import os

from flask import render_template

os.makedirs('output/templates', exist_ok = True)
app = flask.Flask(__name__, template_folder='output/templates', static_folder="output")

@app.route("/")
def index():
    directories = os.listdir('output/templates')

    table_rows = ""

    for output in directories:
        table_rows += f'<a href=/page/{output[:-5]}>{output[:-5]}</a><br>'

    return table_rows

@app.route('/page/<video>')
def test(video):
    return render_template(f'{video}.html')

app.run(debug=True)