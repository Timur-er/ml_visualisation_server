from flask import Flask
from flask_socketio import SocketIO
from sklearn import datasets
from sklearn.model_selection import train_test_split
from algorithms.Perceptron.Perceptron import Perceptron

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='http://localhost:3000')

perceptron = Perceptron()


@app.route('/')
def hello_world():
    return 'Hello World! Lets start building ai and ml app'


@app.route('/train', methods=["POST"])
def train():
    return 'Endpoint for train model'


@socketio.on('start_training')
def handle_training():
    x, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    training_data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist()}
    socketio.emit('initial_data', training_data)

    for weights, bias in perceptron.fit(x_train, y_train):
        data = {'weights': weights.tolist(), 'bias': bias}
        socketio.emit('update_data', data)
        socketio.sleep(0.3)


if __name__ == '__main__':
    app.run()
