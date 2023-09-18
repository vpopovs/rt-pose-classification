import tflite_runtime as tflite
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import time
import threading

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

POSE_NAMES = ["Sitting", "Standing arms down", "Standing arms up", "T-pose"]

def draw_prediction_on_image(
        image, keypoints_with_scores, predicted_label
    ):

    # all keypoints [[[ [x0 y0 confidence0], [x1 y1 confidence1], ... ]]]
    keypoints = keypoints_with_scores[0][0]

    for kpi_1, kpi_2 in KEYPOINT_EDGE_INDS_TO_COLOR:
        
        # if either of keypoints have low confidence, dont draw a line
        if keypoints[kpi_1][2] < 0.1 or keypoints[kpi_2][2] < 0.1:
            continue
            
        kp_loc_1 = (
            int(keypoints[kpi_1][0]),
            int(keypoints[kpi_1][1])
        )    

        kp_loc_2 = (
            int(keypoints[kpi_2][0]),
            int(keypoints[kpi_2][1])
        )    
        
        image = cv2.line(image, kp_loc_1, kp_loc_2, (0, 255, 0), 2)
    
    image = cv2.putText(
        image, 
        POSE_NAMES[predicted_label],
        (0, 150),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 255, 0),
        2
    )
  
    return image

# load the model
interpreter_keypoints = Interpreter(model_path="./project_pose_classifier/keypoints_model.tflite")
interpreter_keypoints.allocate_tensors()

interpreter_classifier = Interpreter(model_path="./project_pose_classifier/classifier_model.tflite")
interpreter_classifier.allocate_tensors()

webcam_capture = cv2.VideoCapture("./lady_standing.mp4")

RESOLUTION = (720, 1280)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./demos/demo-' + str(time.time()) +
                      '.mp4', fourcc, 7, RESOLUTION)

keypoints_with_scores = None
predicted_label = None
video_frame = None
ready_frame = None

stop = False

i = 0

def get_keypoints(image):
    """
    Applies a movenet model to an vertical image of size 720 (w) by 1280 (h).

    Returns a numpy array of shape (1, 51) of 
    each keypoint coordinate and confidence.
    """
    # run the model
    input_details = interpreter_keypoints.get_input_details()
    output_details = interpreter_keypoints.get_output_details()

    interpreter_keypoints.set_tensor(input_details[0]['index'], image)
    interpreter_keypoints.invoke()

    keypoints_with_scores = interpreter_keypoints.get_tensor(output_details[0]['index'])

    keypoint_coords = keypoints_with_scores.reshape(17, 3)
    keypoint_coords[:, 0] *= 1280
    keypoint_coords[:, 1] *= 720
    keypoint_coords[:, [1, 0]] = keypoint_coords[:, [0, 1]]

    return keypoints_with_scores, np.array([keypoint_coords.flatten()])


def predict(keypoints):
    """
    Applies the prediction model to a numpy 
    array of keypoints of shape (1, 51).

    Returns the predicted integer label of the 
    most probable pose represented by the given keypoints.
    """
    input_index = interpreter_classifier.get_input_details()[0]["index"]
    output_index = interpreter_classifier.get_output_details()[0]["index"]

    interpreter_classifier.set_tensor(input_index, keypoints)
    interpreter_classifier.invoke()

    output = interpreter_classifier.tensor(output_index)
    predicted_label = np.argmax(output()[0])

    return predicted_label


def model():
    global ready_frame, stop, video_frame, out, i, keypoints_with_scores, predicted_label
    start_time = time.time()

    while i < 200:
        while ready_frame is None and not stop:
            pass
        
        model_frame = ready_frame
        ready_frame = None

        keypoints_with_scores, keypoints_flat = get_keypoints(model_frame)
        predicted_label = predict(keypoints_flat)

        i += 1
        # print(i)

    stop = True
    out.release()
    cv2.destroyAllWindows()
    print(f"Source fps: {i / (time.time() - start_time)}")


def main():
    global ready_frame, stop, video_frame, keypoints_with_scores, predicted_label
    while not stop:
        if predicted_label is not None:
            output_image = draw_prediction_on_image(
                video_frame, keypoints_with_scores, predicted_label
            )

            output_image = cv2.resize(output_image, RESOLUTION)
            out.write(output_image)
            keypoints_with_scores = None
            predicted_label = None

        _, frame = webcam_capture.read()
        video_frame = frame
        image = frame

        # convert rgba (4 channels) to rgb (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
        # resize image
        dim = (192, 192)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
        resized = np.expand_dims(resized, axis=0)

        # wait for prev frame to be consumed
        while ready_frame is not None and not stop:
            pass

        ready_frame = resized


t_main = threading.Thread(target=main)

start_time = time.time()
t_main.start()

model()
