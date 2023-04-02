import sys
import argparse
import os
import json
import flask

import cv2
import numpy as np
import threading
import queue
import ast


BUF_SIZE = 100
q = queue.Queue(BUF_SIZE)

class KeyFrames(object):

    def __init__(self, server, json_hash, image_file, kf_dir, method, dist, cropping):
        [self.server, self.json_hash, self.image_file, self.keyframes_path, self.method, self.dist, self.cropping] = \
            [server, json_hash, image_file, kf_dir, method, dist, cropping]

        if not os.path.isdir(self.keyframes_path):
            print("Cannot open key frames directory", self.keyframes_path)
            exit()

        if self.method == "average":
            self.hash = cv2.img_hash.AverageHash_create()
        if self.method == "phash":
            self.hash = cv2.img_hash.PHash_create()
        if self.method == "color":
            self.hash = cv2.img_hash.ColorMomentHash_create()

        print("json_hash: ", json_hash)
        print("Key frames directory: ", kf_dir)

        if self.json_hash != None:
            self.key_filename = os.path.basename(self.json_hash)
            self.key = os.path.splitext(self.key_filename)[0]
            with open(self.json_hash) as json_file:
                file_data = json.load(json_file)
            self.frame_hash = np.asarray(file_data[self.key][0]['hash'], dtype=np.uint8)

        self.cropping_val = round(self.cropping / 100, 2)

        if self.image_file != None:
            img = cv2.imread(self.image_file)
            img_cropped = self.__cropping__(img)
            # frame = cv2.resize(img_cropped, (8, 8), 0, 0, cv2.INTER_AREA)
            self.frame_hash = self.hash.compute(img_cropped)

    def __del__(self):
        print("Freeing resources")
        print("Terminate")

    def __cropping__(self, img):
        height = img.shape[0]
        width = img.shape[1]
        img_cropped = img[int(height * self.cropping_val):int(height - height * self.cropping_val), 0:width]
        return img_cropped

    @staticmethod
    def hamming(a, b):
        return bin(int(a) ^ int(b)).count('1')

    def distance(self, hash1, hash2):
        return self.hash.compare(hash1, hash2)

    def locate(self):
        video_file = None

        for root, dirs, files in os.walk(self.keyframes_path):
            for file in files:
                if not file.endswith('.json'):
                    continue
                with open(self.keyframes_path + "/" + file) as json_file:
                    data = json.load(json_file)

                key = os.path.splitext(file)[0]
                hits = [self.distance(np.asarray(data[key][i]['hash'], dtype=np.uint8), self.frame_hash) <= self.dist
                        for i in range(len(data[key]))]
                if sum(hits) >= 1:
                    print(self.keyframes_path + "/" + file, " video hit!")
        return video_file

    def locate_from_JSONfile(self, android_JSON_data):
        video_file = None

        for root, dirs, files in os.walk(self.keyframes_path):
            for file in files:
                if not file.endswith('.json'):
                    continue
                with open(self.keyframes_path + "/" + file) as json_file:
                    data = json.load(json_file)

                # key of the video hash file
                key = os.path.splitext(file)[0]

                # key of the android_JSON_data recevied
                mobile_id = list(android_JSON_data.keys())[0]
                # For each hash value inside the android JSON data
                for i in range(114, len(android_JSON_data[mobile_id])):
                    hash_str = android_JSON_data[mobile_id][i]['hash']
                    hash_val = np.array(ast.literal_eval(hash_str), dtype=np.uint8)

                    # hits of the hash_val inside the video hash file
                    hits = [self.distance(np.asarray(data[key][i]['hash'], dtype=np.uint8), hash_val) <= self.dist
                        for i in range(len(data[key]))]

                    if sum(hits) >= 1:
                        print(self.keyframes_path + "/" + file, " video hit!")
        return video_file


def check_args(args=None):
    parser = argparse.ArgumentParser(description='Video process')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json_hash", help="json that contains hash")
    group.add_argument("--image_file", help="image file to check")
    group.add_argument("--server", action='store_true', help="launch server to receive ")
    parser.add_argument("--keyframes_path", help="path of key frames", required=True)
    parser.add_argument('--distance', type=int, help='image distance', default=0)
    parser.add_argument('--method', help='hash method', default='average', choices=['average', 'phash', 'color'])
    parser.add_argument('--cropping', type=int,
                        help='vertical cropping percentaje over the image (do not type percentaje sign here) ',
                        default=0)

    results = parser.parse_args(args)

    if not (0 <= results.cropping < 50):
        results.cropping = 33

    return results.server, results.json_hash, results.image_file, results.keyframes_path, results.method, results.distance, results.cropping

def crop_image(img, val):
    height = img.shape[0]
    width = img.shape[1]
    img_cropped = img[int(height * val):int(height - height * val), 0:width]
    return img_cropped


class processJSON(threading.Thread):
    def __init__(self, data):
        super(processJSON, self).__init__()
        self.data = data

    def run(self):
        print("Procesando JSON")
        print(self.data)


def main(server, json_hash, image_file, kf_path, hash_method, hash_dist, cropping):
    keyframe = KeyFrames(server, json_hash, image_file, kf_path, hash_method, hash_dist, cropping)
    if (server == None):
        keyframe.locate()
    else:
        app = flask.Flask(__name__)

        @app.route('/jsonreq', methods=['POST'])
        def jsonreq():
            global data  # just here so I can print it from the console for testing purposes.
            data = flask.request.get_json()
            keyframe.locate_from_JSONfile(data)

            return "JSON Read"

        app.run(host="192.168.1.23", port=8080, debug=True)

    return 0


if __name__ == '__main__':
    server, json_hash, image_file, keyframes_path, method, distance, cropping = check_args(sys.argv[1:])
    sys.exit(main(server,json_hash, image_file, keyframes_path, method, distance, cropping))

    """img1 = cv2.imread("frame_movil3.png")
    img1 = crop_image(img1, .33)
    img2 = cv2.imread("145_193_78_1_1_4_167_247__frame95_average.jpg")
    img2 = crop_image(img2, .33)
    #img1 = cv2.resize(img1, (180, 230))

    hasher = cv2.img_hash.AverageHash_create()
    hash1 = hasher.compute(img1)
    hash2 = hasher.compute(img2)

    print(hasher.compare(hash1, hash2))

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey()"""

    """frame_hash = np.zeros(hash1.shape, dtype=np.uint8)
    frame_hash[0] = [145, 193,  78,   1,   1,   4, 167, 247]
    print(hasher.compare(hash1, frame_hash))"""



