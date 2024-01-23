import cv2
import torch
from ultralytics import YOLO
import multiprocessing


def webcam_capture(output_queue):
    cap_webcam = cv2.VideoCapture(0)  # 웹캠 캡처 객체

    while True:
        ret, frame = cap_webcam.read()
        if not ret:
            break
        output_queue.put(frame)  # 웹캠 프레임을 큐에 삽입

    cap_webcam.release()

def video_playback(local_video_path, output_queue):
    cap_video = cv2.VideoCapture(local_video_path)  # 로컬 비디오 캡처 객체

    while True:
        ret, frame = cap_video.read()
        if not ret:
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오 처음으로 되돌리기
            continue
        output_queue.put(frame)  # 로컬 비디오 프레임을 큐에 삽입

    cap_video.release()


# Default blur value
blur = 0
def get_person_mask(frame, result):
    """
    Function attempts to detect
    segmentation mask of a person
    on current video frame
    and returns it
    """

    # get all detected persons on the frame
    classes = [index for (index, cls) in enumerate(result.boxes.cls.numpy()) if cls == 0]
    if len(classes) == 0:
        return

    # get segmentation mask of the first detected person
    index = classes[0]
    mask = result.masks.data[index].numpy().astype('uint8')
    # resize to the size of frame, if needed
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    return mask


def apply_background_options(frame, mask, bg):
    """
    Function apply new background and
    or blur for all pixels of specified
    frame except the pixels, that included
    in the person mask 
    """

    bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

    if blur > 0:
        bg = set_blur(bg, blur)    

    # change all pixels of the frame
    # except pixels included to the person mask
    # to pixels from background
    if mask is not None:
        frame[mask == 0] = bg[mask == 0]
    else:
        frame = bg
    return frame


def set_blur(frame, value):
    """
    Function applies blur to specified frame
    """
    if value == 0:
        return frame
    return cv2.blur(frame, (value, value))

if __name__ == "__main__":
    model = YOLO('yolov8n-seg.pt').to(torch.device("mps"))
    # 큐 생성
    queue_webcam = multiprocessing.Queue(maxsize=1)
    queue_video = multiprocessing.Queue(maxsize=1)

    # 프로세스 생성 및 시작
    process_webcam = multiprocessing.Process(target=webcam_capture, args=(queue_webcam,))
    process_video = multiprocessing.Process(target=video_playback, args=("/Users/jaeung/Movies/4K Video Downloader/ninimo.mp4", queue_video,))

    process_webcam.start() 
    process_video.start()

    try:
        while True:
            # 프레임을 큐에서 가져오기
            frame_webcam = queue_webcam.get()
            frame_video = queue_video.get()
            results = model(frame_webcam)
            result = results[0].to('cpu')
            frame_webcam_resized = cv2.resize(frame_webcam, (frame_video.shape[1], frame_video.shape[0]))
            mask = get_person_mask(frame_webcam, result)
            frame = apply_background_options(frame_webcam, mask, frame_video)
            cv2.imshow("YOLOv8 Inference", frame)
            if cv2.waitKey(1) == 27:  # 'Esc' 키 눌림을 감지하여 종료
                break
    finally:
        # 자원 정리
        process_webcam.terminate()
        process_video.terminate()
        process_webcam.join()
        process_video.join()
        cv2.destroyAllWindows()