import argparse
import cv2
import numpy as np
import os

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-H", "--heigh", default=128, type=int,
    	help="heigh of the output")
    ap.add_argument("-W", "--width", default=128, type=int,
    	help="width of the output")
    ap.add_argument("-d", "--dir", default="lfw_funneled",
    	help="path to dir with images")
    ap.add_argument("-x", "--modelxml", default="models/haarcascade_frontalface_default.xml",
    	help="path to haarcascade xml file")
    return vars(ap.parse_args())

def main():
    args = get_args()

    print("[INFO] loading model...")
    face_cascade = cv2.CascadeClassifier(args['modelxml'])

    print("[INFO] detecting faces...")
    for person_name in os.listdir(args["dir"]):
        person_name_dir = os.path.join(args["dir"], person_name)
        detected_face = False

        if not os.path.isdir(person_name_dir):
            continue

        for picture in os.listdir(person_name_dir):
            person_name_pic = os.path.join(args["dir"], person_name, picture)

            if not os.path.isfile(person_name_pic):
                continue

            img = cv2.imread(person_name_pic)

            biggest_face = None
            quality = 9
            while biggest_face is None:
                faces = face_cascade.detectMultiScale(img, 1.1, quality)

                for face in faces:
                    if biggest_face is None or biggest_face[2] > face[2]:
                        biggest_face = face

                if quality == 2:
                    break
                quality -= 1

            if biggest_face is None:
                print(f"[INFO] Could not detect a face in {person_name_pic}")
                os.remove(person_name_pic)
                continue

            detected_face = True
            (x, y, w, h) = biggest_face

            cropped = img[y:y+h, x:x+w]
            resized = cv2.resize(cropped, (args["heigh"], args["width"]), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(person_name_pic, resized)
            # cv2.imshow('Face', resized)
            # cv2.waitKey()

        if not detected_face:
            print(f"[INFO] Could not detect a face in any image for {person_name_dir}")
            os.rmdir(person_name_dir)

    print("[INFO] finished...")

main()
