from ultralytics import YOLO

model = YOLO('runs/detect/seat_detector-3/weights/best.pt')

results = model.predict(
    source='test7.png',
    conf=0.5,
    save=True,
    show=True
)