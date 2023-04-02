# pip install opencv-python
# pip install opencv-contrib-python


import cv2

def hamming(a, b):
    """
    Calcula la distancia Hamming entre a y b.
    """
    return bin(int(a) ^ int(b)).count('1')

def dhash(image, hash_size=8):
    """
    Calcula el dhash de la imagen de entrada.
    :param image: Imagen a la cuaal le calcularemos el dhash.
    :param hash_size: Número de bytes en el hash resultante.
    """
    # Resdimensionamos la imagen con base al tamaño del hash.
    resized = cv2.resize(image, (hash_size + 1, hash_size))

    # Generamos la imagen de diferencias de píxeles adyacentes.
    diff = resized[:, 1:] > resized[:, :-1]

    # Calculamos el hash.
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])


# https://stackoverflow.com/questions/55685803/using-opencvs-image-hashing-module-from-python
# pip install opencv-python
# pip install opencv-contrib-python
def pHash(cv_image):
    imgg = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY);

    #hsh = cv2.img_hash.BlockMeanHash_create()
    #hsh.compute(a_1)

    h = cv2.img_hash.pHash(imgg)  # 8-byte hash
    pH = int.from_bytes(h.tobytes(), byteorder='big', signed=False)
    return pH


def key_frame_extractor(file):
    import cv2
    import os
    from matplotlib import pyplot as plt
    import numpy as np
    cap = cv2.VideoCapture(file)
    frame_list = []
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print("Error cant make directories")

    cframe = 0
    while (True):
        if os.path.exists('data'):
            break

        ret, frame = cap.read()
        name = './data/' + str(cframe) + '.jpg'
        print("creating" + name)
        cv2.imwrite(name, frame)

        frame_list.append(frame)
        cframe += 1

        if not ret:
            break

    images = {}
    index = {}

    import glob

    for imagePath in glob.glob('./data/*.jpg'):
        filename = imagePath[imagePath.rfind("/") + 1:]

        image = cv2.imread(imagePath, 1)
        images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        index[filename] = hist

    OPENCV_METHODS = (
        (cv2.HISTCMP_CORREL),
        (cv2.HISTCMP_CHISQR),
        (cv2.HISTCMP_INTERSECT),
        (cv2.HISTCMP_BHATTACHARYYA))

    for method in OPENCV_METHODS:

        results = {}
        reverse = False

        if method in (cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT):
            reverse = True

    for (k, hist) in index.items():
        d = cv2.compareHist(index[k], hist, cv2.HISTCMP_INTERSECT)
        results[k] = d
        print(d)

    for (k, hist) in index.items():
        mean__ = np.mean(index[k], dtype=np.float64)

    for (k, hist) in index.items():
        variance = np.var(index[k], dtype=np.float64)

    print("variance", variance)

    standard_deviation = np.sqrt(variance)
    th = mean__ + standard_deviation + 2
    print("threshold value", th)

    try:
        if not os.path.exists('keyframes'):
            os.makedirs('keyframes')
    except OSError:
        print("Error cant make directories")

    cframe1 = 0
    for (k, hist) in index.items():
        d = cv2.compareHist(index[k], hist, cv2.HISTCMP_INTERSECT)
        ret, keyframe = cap.read()

        if not ret:
            break

        if (d > th):
            name = './keyframes/' + str(cframe1) + '.jpg'
            print("creating" + name)
            cv2.imwrite(name, keyframe)
            cframe1 += 1



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # read video from file
    #VIDEO_FILE = "/Users/luismg/PycharmProjects/SmartMe/cnc_vol_coralesoeraantes_15_comisiones.mp4"
    VIDEO_FILE = "/Users/luismg/PycharmProjects/SmartMe/cnc_vol_kathecreadoracontenido_15_comisiones.mp4"

    #key_frame_extractor(VIDEO_FILE)

    cap = cv2.VideoCapture(VIDEO_FILE)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 202)
    ret,frame1 = cap.read()
    #frame1 = cv2.resize(frame1, (320, 640))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 203)
    ret,frame2 = cap.read()
    #frame2 = cv2.resize(frame2, (640, 320))

    frame1 = cv2.imread('/Users/luismg/PycharmProjects/SmartMe/keyframes/879675986540029584060318323791474615116605948860653632056.jpg')
    frame2 = cv2.imread('/Users/luismg/PycharmProjects/SmartMe/movil2.png')
    width = frame2.shape[1]
    height = frame2.shape[0]
    cropping_val = 0.33
    frame2 = frame2[int(height * cropping_val):int(height - height * cropping_val), 0:width]

    cv2.imshow("frame1", frame1)
    cv2.imshow('frame2', frame2)

    first_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    first_image_hash = dhash(first_image)

    second_image = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    second_image_hash = dhash(second_image)

    # Computamos la distancia Hamming entre ambos hashes.
    distance = hamming(first_image_hash, second_image_hash)
    print ('distamcia 1:',distance)

    if distance < 10:
        print('Las imágenes son perceptualmente IGUALES.')
    else:
        print('Las imágenes son perceptualmente DIFERENTES.')

    ph1 = pHash(frame1)
    ph2 = pHash(frame2)
    distance2 = hamming(ph1, ph2)
    print ('distancia 2:',distance2)
    if distance2 < 10:
        print('Las imágenes son perceptualmente IGUALES.')
    else:
        print('Las imágenes son perceptualmente DIFERENTES.')


    # Usando las librerias de opencv2
    # f1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # f1_gray_sized = cv2.resize(f1_gray, (8,8), 0, 0,  cv2.INTER_LINEAR_EXACT)
    # f2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # f2_gray_sized = cv2.resize(f2_gray, (8, 8), 0, 0, cv2.INTER_LINEAR_EXACT)

    #hsh = cv2.img_hash.BlockMeanHash_create()
    hsh = cv2.img_hash.AverageHash_create()

    cv2.imshow("f1", frame1)
    cv2.imshow("f2", frame2)
    cv2.waitKey()

    h1 = hsh.compute(frame1)
    h2 = hsh.compute(frame2)
    diff = hsh.compare(h1,h2)
    print("Image hamong distance is: ", diff)

    # hashval1 = imagehash.phash(frame1)
    # hashval2 = imagehash.phash(frame2)

    cv2.waitKey()

    while True:
        ret, frame = cap.read()
        if not ret: break  # break if no next frame

        cv2.imshow("Video",frame)  # show frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # on press of q break
            break

    # release and destroy windows
    cap.release()
    cv2.destroyAllWindows()
