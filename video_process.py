import sys
import argparse
import cv2
from tqdm import tqdm
import os
import json



class Video (object):

    def __init__(self, src, dst, dist, meth, crop, frame, videoframes):
        [self.video_source, self.destination_path, self.dist, self.method, self.cropping, self.frame_n, self.videoframes] = \
            [src, dst, dist, meth, crop, frame, videoframes]

        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("Cannot open video source", self.video_source)
            exit()

        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.frame_n is not None:
            if (self.frame_n > self.length):
                self.frame_n = self.length

        self.frames = []
        self.key_frames = []
        self.cropping_val = round(self.cropping / 100,2)

        if meth == "average":
            self.hash = cv2.img_hash.AverageHash_create()
        if meth == "phash":
            self.hash = cv2.img_hash.PHash_create()
        if meth == "color":
            self.hash = cv2.img_hash.ColorMomentHash_create()




        print ("Video source: ", src)
        print ("Destination path: ", dst)
        print ("Method:", meth)
        print ("Distance: ", dist)
        print ("Cropping %: ", self.cropping)

    def __del__(self):
        print("Freeing resources")

        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

        print("Terminate")

    def distance(self, hash1, hash2):

        return self.hash.compare(hash1, hash2)


    def load_video(self):
        for i in tqdm(range(self.length), desc="Loading video"):
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_crp = frame[int(self.height * self.cropping_val):int(self.height - self.height * self.cropping_val),0:self.width]
            #frame = cv2.resize(frame_crp, (8, 8), 0, 0, cv2.INTER_AREA)

            hash_val = self.hash.compute(frame_crp)

            frame_dict = {
                "frame": frame_crp,
                "n_frame" : i,
                "hash": hash_val
            }
            self.frames.append(frame_dict)

    def get_key_frames(self):
        return self.key_frames

    def get_frames(self):
        return self.frames

    @staticmethod
    def show_video(frames):
        for frame in tqdm(list(frames), desc="Showing video"):
            cv2.imshow("Video", frame['frame'])  # show frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # on press of q break
                break

    def process_video(self):
        if self.frame_n is not None:
            self.key_frames.append(self.frames[self.frame_n])
            return

        [i,n] = [0, self.length]

        pbar = tqdm(total = self.length, desc="Processing video")
        while i < n:
            first = self.frames[i]
            #cv2.imshow("first", first["frame"])
            i += 1
            incr = 1
            for j in range(i, n):
                second = self.frames[j]
                if  self.distance(first['hash'], second['hash']) >= (self.dist - 1):
                    #print ("Almacenando frame ",i-1)
                    #cv2.imshow("almacenado", second["frame"])
                    self.key_frames.append(first)
                    incr = j-i+1
                    i=j
                    #cv2.waitKey()
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):  # on press of q break
                break
            pbar.update(incr)

        pbar.close()


    def save_key_frames(self):
        for key_frame in tqdm(list(self.key_frames), desc="Writing files"):
            name = self.destination_path + '/' + str(key_frame['hash']) + '.jpg'
            cv2.imwrite(name, key_frame['frame'])

# Hacer JSON especiifcando nombre de fichero con los campos: hash, method, distance

    def write_key_frames_hashes(self):
        file_name = os.path.basename(self.video_source)
        data = {}
        data.setdefault(file_name)
        r = list()
        for key_frame in tqdm(list(self.key_frames), desc="Writing files"):
            r.append({'n_frame':key_frame['n_frame'], 'hash':key_frame['hash'].tolist(),'hash_method':self.method,'distance:':self.dist,'cropping':self.cropping})
            if self.videoframes:
                name = self.destination_path + '/' + str(key_frame['hash']) + '.jpg'
                cv2.imwrite(name, key_frame['frame'])
        data[file_name] = r

        with open(self.destination_path + "/" + file_name + ".json", 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def check_hits(self):
        hits = 0
        for frame in tqdm(list(self.frames), desc="Checking hits"):
            for key_frame in (list(self.key_frames)):
                if self.distance(frame['hash'], key_frame['hash']) <= self.dist:
                    hits += 1
                    break
        print ("Hits: ", hits, "/", self.length)




def check_args(args=None):
    parser = argparse.ArgumentParser(description='Video process')
    parser.add_argument("--source", help="video source", required=True)
    parser.add_argument("--destination_path", help="path for storing the frames", required=True)
    parser.add_argument('--distance', type=int, help='image distance', default=0)
    parser.add_argument('--method', help='hash method', default='average', choices=['average', 'phash', 'color'])
    parser.add_argument('--cropping', type=int, help='vertical cropping percentaje over the image (do not type percentaje sign here) ', default=33)
    parser.add_argument('--frame', type=int, help='frame number to get the hash')
    parser.add_argument('--videoframes', help='writes video frames', action='store_true')


    results = parser.parse_args(args)
    if not (results.cropping >= 0 and results.cropping < 50):
        results.cropping = 33

    if results.frame is not None:
        if not (results.frame >= 0 and results.frame < 777):
            0 if results.frame < 0 else 777


    return results.source, results.destination_path, results.distance, results.method, results.cropping, results.frame, results.videoframes


def main(src, dst, dist, meth, crop, frame, videoframes):
    video = Video(src, dst, dist, meth, crop, frame, videoframes)
    video.load_video()

    video.process_video()
    video.check_hits()
    video.write_key_frames_hashes()

    video.show_video(video.get_key_frames())
    #cv2.waitKey()
    return 0



if __name__ == '__main__':
    source, destination, distance, method, crop, frame, videoframes = check_args(sys.argv[1:])
    sys.exit(main(source, destination, distance, method, crop, frame, videoframes))

