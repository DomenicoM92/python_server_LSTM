from keras.models import model_from_json
import numpy as np
import LSTM_Keras
from flask import Flask, render_template
# set the project root directory as the static folder, you can set others.
app = Flask(__name__)
input = None
xml_sample = []
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train")
def train():
    LSTM_Keras.run_experiment()

@app.route("/classify")
def classify():
    input = get_input()
    # # load json and create model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    prediction = loaded_model.predict(input, verbose=1)
    prediction = np.array(prediction)
    index_max = np.argmax(prediction)
    print(index_max)
    print(prediction)
    print("Motion Class: "+ return_class(index_max))

    return render_template("index.html")

@app.route("/sensor_stream")
def sensor_stream():
    import socket
    import lxml.etree
    global xml_sample
    numb = 1
    port = 7059
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", port))
    print("waiting on port:", port)
    while 1:
        data, addr = s.recvfrom(1024)
        print("Sensor sample number: "+ str(numb))

        print(data.decode("utf-8"))
        xml_sample.append(data.decode("utf-8"))
        numb+=1
        if numb == 129:
            print("END first sample")
            print(xml_sample)
            break

    from xml.etree import ElementTree as ET
    acc_x = ""
    acc_y = ""
    acc_z = ""
    gyro_x = ""
    gyro_y = ""
    gyro_z = ""
    tot_acc_x = ""
    tot_acc_y = ""
    tot_acc_z = ""
    for xml in xml_sample:
        well_formed_xml = "<root><NodeId>" + xml.split("<NodeId>")[1] + "</root>"
        print(well_formed_xml)
        root = ET.fromstring(well_formed_xml)
        print(root.findall('Accelerometer/Accelerometer1')[0].text)
        acc_x = acc_x + "  " + root.findall('Accelerometer/Accelerometer1')[0].text
        acc_y = acc_y + "  " + root.findall('Accelerometer/Accelerometer2')[0].text
        acc_z = acc_z + "  " + root.findall('Accelerometer/Accelerometer3')[0].text
        gyro_x = gyro_x + "  " + root.findall('Gyroscope/Gyroscope1')[0].text
        gyro_y = gyro_y + "  " + root.findall('Gyroscope/Gyroscope2')[0].text
        gyro_z = gyro_z + "  " + root.findall('Gyroscope/Gyroscope3')[0].text
        tot_acc_x = tot_acc_x + "  " + root.findall('LinearAcceleration/LinearAcceleration1')[0].text
        tot_acc_y = tot_acc_y + "  " + root.findall('LinearAcceleration/LinearAcceleration2')[0].text
        tot_acc_z = tot_acc_z + "  " + root.findall('LinearAcceleration/LinearAcceleration3')[0].text
    print(acc_x)
    with open("data/stream_data/Inertial Signals/body_acc_x_test.txt", "w") as text_file:
        text_file.write("%s" % acc_x)
    with open("data/stream_data/Inertial Signals/body_acc_y_test.txt", "w") as text_file:
        text_file.write("%s" % acc_y)
    with open("data/stream_data/Inertial Signals/body_acc_z_test.txt", "w") as text_file:
        text_file.write("%s" % acc_z)
    with open("data/stream_data/Inertial Signals/body_gyro_x_test.txt", "w") as text_file:
        text_file.write("%s" % gyro_x)
    with open("data/stream_data/Inertial Signals/body_gyro_y_test.txt", "w") as text_file:
        text_file.write("%s" % gyro_y)
    with open("data/stream_data/Inertial Signals/body_gyro_z_test.txt", "w") as text_file:
        text_file.write("%s" % gyro_z)
    with open("data/stream_data/Inertial Signals/total_acc_x_test.txt", "w") as text_file:
        text_file.write("%s" % tot_acc_x)
    with open("data/stream_data/Inertial Signals/total_acc_y_test.txt", "w") as text_file:
        text_file.write("%s" % tot_acc_y)
    with open("data/stream_data/Inertial Signals/total_acc_z_test.txt", "w") as text_file:
        text_file.write("%s" % tot_acc_z)
    s.close()
    return render_template("index.html")


def return_class(index):
    if index == 0:
        return "WALKING"
    elif index == 1:
        return "WALKING UPSTAIRS"
    elif index == 2:
        return "WALKING DOWNSTAIRS"
    elif index == 3:
        return "SITTING"
    elif index == 4:
        return "STANDING"
    elif index == 5:
        return "LAYING"

def get_input():
    import numpy as np
    def load_X(X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]
    # FETCH DATASET

    DATA_PATH = "data/"
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    DATASET_PATH = DATA_PATH + "stream_data/"
    print("\n" + "Stream data are located located at: " + DATASET_PATH)

    # PREPAIRING DATASET

    X_test_signals_paths = [
        DATASET_PATH + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]

    X_test_new = load_X(X_test_signals_paths)
    print(X_test_new)
    return X_test_new


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)

