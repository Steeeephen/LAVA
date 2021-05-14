import flask
import os
import threading

from flask import render_template, request, send_from_directory, redirect

from assets.lava import LAVA

app = flask.Flask(__name__, template_folder='assets/webui_templates', static_folder="output")

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/games')
def games():
  directories = os.listdir('output')

  directories.remove('positions')

  return render_template('games.html', games=directories)

@app.route('/data')
def data():
  games = os.listdir('output/positions')
  games = [game.split('.')[0] for game in games]

  return render_template('data.html', games=games)

@app.route('/download_data/<game>')
def download_data(game):
  positions_directory = os.path.join(
    'output',
    'positions',
    game)

  return send_from_directory(positions_directory, game, as_attachment=True)

@app.route('/delete_data/<game>')
def delete_data(game):
  positions_file = os.path.join(
    'output',
    'positions',
    game)

  os.remove(positions_file)

  games = os.listdir('output/positions')
  games = [game.split('.')[0] for game in games]

  return redirect('/data')

@app.route("/input", methods=['GET', 'POST'])
def input():
  if request.method=='GET':
    return render_template('input.html')
  elif request.method == 'POST':
    form = request.form
    
    lava = LAVA()

    url = form['url']
    local = form.get('local', False) == 'true'
    playlist = form.get('playlist', False) == 'true'
    graphs = form.get('graphs', False) == 'true'
    minimap = form.get('minimap', False) == 'true'

    run_video_thread = threading.Thread(
      target=lava.execute, 
      name="tracker", 
      args=(
        url,
        local, 
        playlist, 
        0, 
        minimap,
        graphs
      )
    ).start()

    return render_template('input.html')

@app.route('/games/<video>')
def test(video):
    return render_template('game.html', game=video)

app.run(debug=True, port=8080)