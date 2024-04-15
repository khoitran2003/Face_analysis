import cv2
import argparse
import torch
import yaml
import random
from PIL import Image
from ultralytics import YOLO
from source.classification.data_loader_clf import image_transformation
from source.classification.resnet_50_modify import Modified_Resnet_50

def get_args():
    parsers = argparse.ArgumentParser(description='Face')
    parsers.add_argument(
        '--webcam_resolution',
        default=[608, 342],
        nargs=2,
        type=int
        )
    parsers.add_argument(
        "--cfg",
        type=str,
        default="cfg/classifier.yaml",
    )
    parsers.add_argument(
        "--label_cfg",
        type=str,
        default="cfg/labels.yaml",
    )
    args = parsers.parse_args()
    return args

def find_key(yaml_file, value):
    for key, val in yaml_file.items():
        if val == value:
            return key
colors = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Yellow
    (255, 165, 0),    # Orange
    (255, 192, 203),  # Pink
    (128, 0, 128),    # Purple (replacing with light purple)
    (0, 128, 128),    # Teal (replacing with light teal)
    (255, 255, 0),    # Lime (replacing with light lime)
    (75, 0, 130),     # Indigo (replacing with light indigo)
    (255, 127, 80),   # Coral (replacing with light coral)
    (230, 230, 250)   # Lavender
]

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.cfg) as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with open(args.label_cfg) as label_cfg_file:
        label_cfg = yaml.safe_load(label_cfg_file)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    frame_width, frame_height = args.webcam_resolution
    
    yolomodel = YOLO('checkpoint/detection.pt')
    clfmodel = Modified_Resnet_50()
    checkpoint = torch.load('checkpoint/classifier_best.pt', map_location=device)
    clfmodel.load_state_dict(checkpoint['state_dict'])
    clfmodel.to(device).eval()
    names = yolomodel.names

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    # while cap.isOpened():
    #     flag, frame = cap.read()
    #     frame = cv2.flip(frame, 1)
    #     if not flag:
    #         break
    
    image_path = "/home/khoi/CVProjects/face_analysis/uploads/MESSI.jpg"
    frame = cv2.imread(image_path)
    # print(frame.shape)
    results = yolomodel.predict(frame, imgsz=320)

    xyxys = results[0].boxes.xyxy.cpu().numpy()
    class_names = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    face_list = []
    for (bbox, c, conf) in zip(xyxys, class_names, confs):
        cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        cropped_frame = Image.fromarray(cropped_frame)
        trans_cropped_frame = image_transformation(cropped_frame, cfg)
        face_list.append(trans_cropped_frame)
        if face_list:
            batch_face = torch.stack(face_list, dim=0)
        with torch.no_grad():
            clf_pred = clfmodel(batch_face)
            age, gender, emotion = clf_pred
            age = torch.argmax(age, dim=1)[0]
            gender = torch.argmax(gender, dim=1)[0]
            emotion = torch.argmax(emotion, dim=1)[0]
            
            age = find_key(label_cfg['age'], age)
            gender = find_key(label_cfg['gender'], gender)
            emotion = find_key(label_cfg['emotion'], emotion)
            # name = names[int(c)] + ' ' + f'{conf:.2f}%'
            info = 'khoidepzai'
            (text_width, text_height), _ = cv2.getTextSize(info, font, font_scale, thickness)
        color = random.choice(colors)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
        cv2.rectangle(frame, (int(bbox[2]), int(bbox[1] - 1)), (int(bbox[2] + text_width), int(bbox[1] + 3*text_height+10)),
                    (255, 0, 255), -1)
        cv2.putText(frame, age, (int(bbox[2]), int(bbox[1]+text_height)), font, font_scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)
        cv2.putText(frame, gender, (int(bbox[2]), int(bbox[1]+2*text_height+5)), font, font_scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)
        cv2.putText(frame, emotion, (int(bbox[2]), int(bbox[1]+3*text_height+7)), font, font_scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)


    cv2.imwrite('results/2.jpg', frame)
    # if cv2.waitKey(1) & 0xFF == ('q'):
    #     quit()
# cap.release()
# cv2.destroyAllWindows()
        

if __name__ == '__main__':
    args = get_args()
    main(args)





