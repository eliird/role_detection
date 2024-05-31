import glob
import os
import sys
import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tqdm

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  #annotated_image = np.copy(rgb_image)
  print(face_landmarks_list)
  # Loop through the detected faces to visualize.
  landmarks = []
  for image in face_landmarks_list:
      for landmark in image:
        landmarks.append([landmark.x, landmark.y, landmark.z])

  return rgb_image, np.array(landmarks)

def video2Pose(videoPath, saveFilePath):
    cap = cv2.VideoCapture(0)
    videoLandmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No Frame in Video")
            break
        landmarks = image2Pose(frame)
        videoLandmarks.append(landmarks)
    cap.release()
    videoLadnmarks = np.array(videoLandmarks)
    print(videoLandmarks)
    #np.save(saveFilePath, videoLandmarks)
    return

def image2Pose(frame):

    image = image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)#mp.Image.create_from_file(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_result = detector.detect(image)
    landmarks = draw_landmarks_on_image(detection_result)
    return landmarks
try:
    video2Pose(0, 'x')
except Exception as e:
    print(e)
# try:
   
#     base_path = r"./data"
#     folders = ['video1', 'video2', 'video3']
#     videoPaths = [os.path.join(base_path, i) for i in folders]

#     rec = os.listdir(base_path)
#     rec_only = []
#     for folder in rec:
#         if('20' in folder):
#             rec_only.append(folder)

#     print('______________________')
#     print(rec_only)
#     print('_______________________')

#     for recording in tqdm(rec_only):
#         os.makedirs(recording, exist_ok=True)
#         rec_path = base_path + '/' + recording
#         #print(recording, rec_path)
#         for folder in folders:
#             #print(f"Making Directory {recording}/Pose/{folder}")
#             os.makedirs(recording+'/FacialLandmarks/'+folder, exist_ok=True)
#             file_path = glob.glob(f'{rec_path}/{folder}/*.avi')
#             print(f"Recording: {recording} Person in {folder}")
#             for file in tqdm(file_path):
#                 fileName = file.split('-')[1]
#                 saveFileName = f'{recording}/FacialLandmarks/{folder}/{fileName}.npy'
#                 #print(file, "...", saveFileName)
#                 video2Pose(file, saveFileName)

# except Exception as e:
#     print(e)
#     exc_type, exc_obj, exc_tb = sys.exc_info()
#     print(exc_type, exc_tb.tb_lineno)