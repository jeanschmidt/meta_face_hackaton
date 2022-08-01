from collections import defaultdict
from facenet_pytorch import MTCNN
from scipy import spatial
from skimage import io
from tqdm import tqdm
import argparse
import cv2
import itertools
import math
import numpy as np
import sys
import torch

from .model import APN_Model


MODEL_INPUT_SHAPE = (128, 128)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False,
    	help="video to detect faces")
    ap.add_argument("-i", "--image", required=False,
    	help="image to detect faces")
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
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    faces, scores = face_cascade.detect(img)

    if not faces.shape[0]:
        raise Exception(f"could not detect a face in image {img_path}")

    x, y, xx, yy = [int(n) for n in faces[0]]
    s = max(xx - x, yy - y)
    xx = x + s
    yy = y + s
    cropped = img[y:yy, x:xx]
    resized = cv2.resize(cropped, MODEL_INPUT_SHAPE, interpolation=cv2.INTER_LANCZOS4)
    # cv2.imshow('frame_name', resized)
    # cv2.waitKey()
    return resized


def get_all_faces_image(face_cascade, model, img):
    faces, scores = face_cascade.detect(img)

    ret = []
    for face in faces:
        x, y, xx, yy = int(math.floor(face[0])), int(math.floor(face[1])), int(math.ceil(face[2])), int(math.ceil(face[3])),
        [int(n) for n in face]
        s = max(xx - x, yy - y)
        xx = x + s
        yy = y + s

        cropped = img[y:yy, x:xx]
        if cropped.size < 1:
            continue

        resized = cv2.resize(cropped, MODEL_INPUT_SHAPE, interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow('frame_name', resized)
        # cv2.waitKey()
        ret.append((
            face,
            img_2_vec(model, resized),
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_faces = get_all_faces_image(face_cascade, model, frame)
    faces_scores = []

    for frame_face_idx in range(len(frame_faces)):
        face_coord, face_embedding = frame_faces[frame_face_idx]

        temp_faces = []
        for badge_idx in range(len(args["badges"])):
            dist = spatial.distance.cosine(face_embedding, badges_faces[badge_idx])
            # print(
            #     args["badges"][badge_idx],
            #     dist,
            #     face_embedding[0:5],
            #     badges_faces[badge_idx][0:5],
            # )
            temp_faces.append((2 - dist, args["badges"][badge_idx], face_coord, ))

        total = sum(x[0] for x in temp_faces)
        for inv_dist, badge, face_coord in temp_faces:
            faces_scores.append((1 - (inv_dist / total), badge, face_coord, ))

    faces_scores.sort(key=lambda x: x[0])

    # for score, badge, face_coord in faces_scores:
    #     pos = [int(n) for n in face_coord]
    #     print(score, badge, pos)

    draw_badge = set()
    draw_area = set()
    for score, badge, face_coord in faces_scores:
        (x, y, xx, yy) = [int(n) for n in face_coord]
        face_coord = (x, y, xx, yy, )

        if badge in draw_badge or face_coord in draw_area:
            continue
        draw_badge.add(badge)
        draw_area.add(face_coord)

        yield (score, badge, face_coord, )

def most_likely_face(frames_window, face_idx):
    scores = defaultdict(float)
    for frame in frames_window:
        scores[frame[face_idx][1]] += 1 - frame[face_idx][0]
    return (
        (s, face_idx, b, )
        for b, s in scores.items()
    )

def detect_in_image(badges_faces, face_cascade, model, args):
    print("[INFO] Processing image...")

    image = cv2.imread(args["image"])
    faces = process_one_frame(face_cascade, model, image, args, badges_faces)
    for _, badge, face_coord in faces:
        (x, y, xx, yy, ) = [int(n) for n in face_coord]
        image = cv2.rectangle(image, (x, y, ), (xx, yy, ), (0, 255, 0, ), 2)
        image = cv2.putText(
            image,
            badge,
            (x, yy+15, ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 200),
            4,
            cv2.LINE_AA,
        )
        image = cv2.putText(
            image,
            badge,
            (x, yy+15, ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(args["image"] + ".out.png", image)

def detect_in_video(badges_faces, face_cascade, model, args):
    FRAME_RATE = 60
    cap_in = cv2.VideoCapture(args["video"])
    frame_width = int(cap_in.get(3))
    frame_height = int(cap_in.get(4))
    cap_out = cv2.VideoWriter(args["video"] + ".out.avi", cv2.VideoWriter_fourcc('M','J','P','G'), FRAME_RATE, (frame_width,frame_height))

    print("[INFO] Processing frames...")

    WINDOW_SIZE = 3 * FRAME_RATE
    _, frame = cap_in.read()
    frames_window = [frame, ]
    frames_window_faces = [list(process_one_frame(face_cascade, model, frame, args, badges_faces)), ]
    # max_frames = FRAME_RATE * 3

    print("[INFO] preloading frames...")
    with tqdm(total=WINDOW_SIZE - 1) as pbar:
        for _ in range(WINDOW_SIZE - 1):
            ret, frame = cap_in.read()

            if not ret:
                break

            frames_window.append(frame)
            frames_window_faces.append(list(reorder_closest_points(
                frames_window_faces[-1],
                list(process_one_frame(face_cascade, model, frame, args, badges_faces))
            )))
            pbar.update(1)

    frames_counts = int(cap_in. get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] processing windowed frames...")
    with tqdm(total=frames_counts) as pbar:
        while frames_window_faces:
            ret, new_frame = cap_in.read()
            if ret:
                frames_window.append(new_frame)
                frames_window_faces.append(list(reorder_closest_points(
                    frames_window_faces[-1],
                    list(process_one_frame(face_cascade, model, new_frame, args, badges_faces))
                )))

            curr_frame = frames_window.pop(0)
            curr_frame_faces = frames_window_faces.pop(0)

            detected = {}
            ranked = []
            for face_idx in range(len(curr_frame_faces)):
                ranked += list(most_likely_face(frames_window_faces, face_idx))
            ranked.sort(reverse=True)

            found_idxs = set()
            found_badges = set()
            for score, face_idx, badge in ranked:
                if face_idx in found_idxs or badge in found_badges:
                    continue
                found_idxs.add(face_idx)
                found_badges.add(badge)
                detected[face_idx] = badge

            for face_idx, face_data in enumerate(curr_frame_faces):
                _, _, face_coord = face_data
                (x, y, xx, yy, ) = [int(n) for n in face_coord]
                # badge = most_likely_face(frames_window_faces, face_idx)
                badge = detected[face_idx]
                curr_frame = cv2.rectangle(curr_frame, (x, y, ), (xx, yy, ), (0, 255, 0, ), 2)
                curr_frame = cv2.putText(
                    curr_frame,
                    badge,
                    (x, yy+15, ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 200, 200),
                    4,
                    cv2.LINE_AA,
                )
                curr_frame = cv2.putText(
                    curr_frame,
                    badge,
                    (x, yy+15, ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            cap_out.write(curr_frame)
            pbar.update(1)

            # max_frames -= 1
            # if max_frames < 1:
            #     break

    print(" done!")
    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()

    print("[INFO] loading APN model...")
    model = APN_Model()
    model.load_state_dict(torch.load("models/best_model_0.5.1.pt", map_location=torch.device('cpu')))
    model.eval()

    print("[INFO] loading face model...")
    face_cascade = MTCNN(
        margin=0, post_process=False, select_largest=False,
        selection_method="probability", keep_all=True,
        device=torch.device('cpu')
    )

    print("[INFO] detecting faces in badges...")
    badges_faces = [
        img_2_vec(model, get_best_face_image(face_cascade, i))
        for i in args["badges"]
    ]

    if args["video"]:
        detect_in_video(badges_faces, face_cascade, model, args)

    if args["image"]:
        detect_in_image(badges_faces, face_cascade, model, args)

main()
