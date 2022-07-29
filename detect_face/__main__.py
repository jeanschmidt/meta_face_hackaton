from skimage import io
import argparse
import cv2
import itertools
import numpy as np
import torch
from scipy import spatial

from .model import APN_Model


MODEL_INPUT_SHAPE = (128, 128)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--frames", nargs='+', required=True,
    	help="frames to detect faces")
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


def get_all_faces_image(face_cascade, model, img_path):
    img = cv2.imread(img_path)
    faces = face_cascade.detectMultiScale(img, 1.1, 5)

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

    return (img, ret, )


def print_dbg(args, *pargs, **pkwargs):
    if args["debug"]:
        print(*pargs, **pkwargs)


def main():
    args = get_args()

    print("[INFO] loading APN model...")
    model = APN_Model()
    model.load_state_dict(torch.load("models/best_model_0.1.pt", map_location=torch.device('cpu')))
    model.eval()

    print("[INFO] loading face model...")
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

    print("[INFO] detecting faces in badges...")
    badges_faces = [
        img_2_vec(model, get_best_face_image(face_cascade, i))
        for i in args["badges"]
    ]

    print("[INFO] detecting faces in frames...")
    frames_faces = [
        get_all_faces_image(face_cascade, model, i)
        for i in args["frames"]
    ]

    print("[INFO] scoring faces...")
    best_for_each = {}

    for frame_idx in range(len(args["frames"])):
        frame_name = args["frames"][frame_idx]

        print_dbg(args, f"checking frame... {frame_name} ({len(frames_faces[frame_idx][1])})")

        for frame_face_idx in range(len(frames_faces[frame_idx][1])):
            best = (999999.9, "no face", "no frame", (0, 0, 0, 0, ), )
            print_dbg(args, f"checking face {frame_face_idx}")
            for badge_idx in range(len(args["badges"])):
                face_coord, face_image = frames_faces[frame_idx][1][frame_face_idx]

                dist = spatial.distance.cosine(face_image, badges_faces[badge_idx])
                print_dbg(args, f"    distance for {args['badges'][badge_idx]}: {dist}")
                if dist < best[0]:
                    best = (dist, args["badges"][badge_idx], frame_name, face_coord, )

            print_dbg(args, f"best for face {frame_face_idx} in frame {args['frames'][frame_idx]} is {best[1]} ({best[0]})")
            if best[1] not in best_for_each:
                best_for_each[best[1]] = best
            else:
                if best[0] < best_for_each[best[1]][0]:
                    best_for_each[best[1]] = best

    for badge, best in best_for_each.items():
        print(f"[FOUND] {badge} in frame {best[2]} ({best[0]})")

    for frame_idx in range(len(args["frames"])):
        frame_name = args["frames"][frame_idx]
        img = frames_faces[frame_idx][0]

        for badge, best in best_for_each.items():
            if best[2] == frame_name:
                (x, y, w, h) = best[3]
                img = cv2.rectangle(
                    img,
                    (x, y, ),
                    (x+w, y+h, ),
                    (0, 255, 0),
                    1
                )
                img = cv2.putText(
                    img,
                    best[1],
                    (x, y+h+15, ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        cv2.imshow(frame_name, img)
        cv2.waitKey()


main()
