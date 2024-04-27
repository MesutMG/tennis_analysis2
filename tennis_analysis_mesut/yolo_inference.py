from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('tennis_analysis_mesut/input_videos/input_video.mp4', save=True)
print('---', result, '---')
print('boxes:')
for i in result[0]:
    print(i)