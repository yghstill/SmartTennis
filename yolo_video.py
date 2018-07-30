
from src.yolo import YOLO
from src.yolo import detect_video



if __name__ == '__main__':
    video_path='2018fawang.mp4'
    detect_video(YOLO(), video_path)
