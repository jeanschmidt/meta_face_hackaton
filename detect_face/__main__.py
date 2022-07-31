from scipy import spatial
from skimage import io
import argparse
import cv2
import itertools
import numpy as np
import sys
import torch
from .model import APN_Model
from collections import defaultdict

MODEL_INPUT_SHAPE = (128, 128)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
    	help="video to detect faces")
    ap.add_argument("-b", "--badges", nargs='+', required=True,
    	help="badges as references")
    ap.add_argument('-d', '--debug', default=False, action='store_true',
        help="display help")
    return vars(ap.parse_args())


def img_2_vec(model, img):
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    with torch.no_grad():
        return model(img.unsqueeze(0)).squeeze().numpy()


def get_best_face_image(face_cascade, img_path):
    img = cv2.imread(img_path)

    biggest_face = None
    quality = 9
    while biggest_face is None:
        faces = face_cascade.detectMultiScale(img, 1.1, quality)

        for face in faces:
            if biggest_face is None or biggest_face[2] > face[2]:
                biggest_face = face

        if quality == 1:
            break
        quality -= 1

    if biggest_face is None:
        raise Exception(f"[INFO] Could not detect a face in {img_path}")

    (x, y, w, h) = biggest_face

    cropped = img[y:y+h, x:x+w]
    return cv2.resize(cropped, MODEL_INPUT_SHAPE, interpolation=cv2.INTER_LANCZOS4)


def get_all_faces_image(face_cascade, model, img):
    faces = face_cascade.detectMultiScale(img, 1.1, 3)

    ret = []
    for face in faces:
        (x, y, w, h) = face

        cropped = img[y:y+h, x:x+w]
        # cv2.imshow('frame_name', cropped)
        # cv2.waitKey()
        ret.append((
            face,
            img_2_vec(
                model,
                cv2.resize(cropped, MODEL_INPUT_SHAPE, interpolation=cv2.INTER_LANCZOS4)
            ),
        ))

    return ret


def print_dbg(args, *pargs, **pkwargs):
    if args["debug"]:
        print(*pargs, **pkwargs)


def reorder_closest_points(last, atual):
    for last_frame_face in last:
        one = None
        distance = 99999999999999
        if atual:
            for i, atual_frame_face in enumerate(atual):
                d = (last_frame_face[2][0] - atual_frame_face[2][0]) ** 2 + (last_frame_face[2][1] - atual_frame_face[2][1])
                if d < distance:
                    distance = d
                    one = (atual_frame_face, i, )
            atual.pop(one[1])
            yield one[0]
        else:
            yield last_frame_face

def process_one_frame(face_cascade, model, frame, args, badges_faces):
    frame_faces = get_all_faces_image(face_cascade, model, frame)
    faces_scores = []

    for frame_face_idx in range(len(frame_faces)):
        face_coord, face_embedding = frame_faces[frame_face_idx]

        for badge_idx in range(len(args["badges"])):
            dist = spatial.distance.cosine(face_embedding, badges_faces[badge_idx])
            faces_scores.append((dist, args["badges"][badge_idx], face_coord, ))

    faces_scores.sort(key=lambda x: x[0])

    draw_badge = set()
    draw_area = set()
    for score, badge, face_coord in faces_scores:
        (x, y, w, h, ) = face_coord
        face_coord = (x, y, w, h, )

        if badge in draw_badge or face_coord in draw_area:
            continue
        draw_badge.add(badge)
        draw_area.add(face_coord)

        yield (score, badge, face_coord, )

def most_likely_face(frames_window, face_idx):
    counts = defaultdict(int)
    for frame in frames_window:
        counts[frame[face_idx][1]] += 1
    mx = 0
    badge = ""
    for b, c in counts.items():
        if c > mx:
            mx = c
            badge = b
    return badge

def main():
    args = get_args()

    print("[INFO] loading APN model...")
    model = APN_Model()
    model.load_state_dict(torch.load("models/best_model_0.3.1.pt", map_location=torch.device('cpu')))
    model.eval()

    print("[INFO] loading face model...")
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

    print("[INFO] detecting faces in badges...")
    badges_faces = [
        img_2_vec(model, get_best_face_image(face_cascade, i))
        for i in args["badges"]
    ]

    FRAME_RATE = 19
    cap_in = cv2.VideoCapture(args["video"])
    frame_width = int(cap_in.get(3))
    frame_height = int(cap_in.get(4))
    cap_out = cv2.VideoWriter(args["video"] + "out.avi", cv2.VideoWriter_fourcc('M','J','P','G'), FRAME_RATE, (frame_width,frame_height))

    print("[INFO] Processing frames...")

    WINDOW_SIZE = 3 * FRAME_RATE
    _, frame = cap_in.read()
    frames_window = [frame, ]
    frames_window_faces = [process_one_frame(face_cascade, model, frame, args, badges_faces), ]
    max_frames = FRAME_RATE * 35
    for _ in range(WINDOW_SIZE - 1):
        ret, frame = cap_in.read()

        if not ret:
            break

        sys.stdout.write("|")
        sys.stdout.flush()

        frames_window_faces.append(list(reorder_closest_points(
            frames_window_faces[-1],
            list(process_one_frame(face_cascade, model, frame, args, badges_faces))
        )))

    while frames_window_faces:
        sys.stdout.write("+")
        sys.stdout.flush()

        ret, new_frame = cap_in.read()
        if ret:
            frames_window.append(new_frame)
            frames_window_faces.append(list(reorder_closest_points(
                frames_window_faces[-1],
                list(process_one_frame(face_cascade, model, new_frame, args, badges_faces))
            )))

        curr_frame = frames_window.pop(0)
        curr_frame_faces = frames_window_faces.pop(0)
        for face_idx, face_data in enumerate(curr_frame_faces):
            _, _, face_coord = face_data
            (x, y, _, h, ) = face_coord
            badge = most_likely_face(frames_window_faces, face_idx)
            curr_frame = cv2.putText(
                curr_frame,
                badge,
                (x, y+h+15, ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 200, 200),
                4,
                cv2.LINE_AA,
            )
            curr_frame = cv2.putText(
                curr_frame,
                badge,
                (x, y+h+15, ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cap_out.write(curr_frame)

        max_frames -= 1
        if max_frames < 1:
            break

    print(" done!")
    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()

main()
