import time
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
from itertools import islice
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
import random
import scipy.misc
from scipy.ndimage.measurements import label

print(os.getcwd())

#%%
def undistort(img):
    # Values taken from Project 4 as the same camera is used
    mtx = np.array([
            [1.15396093e+03, 0.00000000e+00, 6.69705359e+02],
            [0.00000000e+00, 1.14802495e+03, 3.85656232e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    dist = np.array([-2.41017968e-01, -5.30720497e-02, -1.15810318e-03, -1.28318544e-04, 2.67124303e-02])
    return cv2.undistort(img, mtx, dist, None, mtx)

for path in islice(glob.glob('C:\projects\CarND\CarND-Advanced-Lane-Lines/camera_cal/*'), 2):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    dst = undistort(img)
    fig, ax = plt.subplots(ncols=2, figsize=(15, 15))
    ax[0].imshow(img)
    ax[1].imshow(dst)

#%%
def extract_hog_features(original_image, visualize=False):
    image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    hog_features = []
    hog_visualized = []

    h = hog(gray,
            orientations=16, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, 
            visualise=visualize, feature_vector=True)        
    
    if (visualize):
        hh = h[0]
        hog_visualized.append(h[1])
    else:
        hh = h
        
    hog_features.extend(hh)

    return (hog_features, hog_visualized)
    

def bin_spatial(original_image):
    resized = cv2.resize(original_image, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)            
    features = hsv.ravel() 
    return features


def color_hist(image):
    channels = []
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    channels.append(np.histogram(hsv[:,:,0], bins=64)[0])
    channels.append(np.histogram(hsv[:,:,2], bins=48)[0])
    channels.append(np.histogram(hsv[:,:,2], bins=32)[0])

    features = np.concatenate(channels)
    return features


def extract_image_features(image):
    spatial_features = bin_spatial(image)
    hist_features = color_hist(np.asarray(image))
    hog_features, hog_visualized = extract_hog_features(image)
    
    #print("Spatial: {}".format(len(spatial_features)))
    #print("Color histogram: {}".format(len(hist_features)))
    #print("HOG: {}".format(len(hog_features)))
    return np.concatenate((spatial_features, hist_features, hog_features))

def extract_features(images):
    return [extract_image_features(i) for i in images]


def load_images(files):
    return [np.asarray(Image.open(f)) for f in files]



#%%
not_cars = extract_features(load_images(random.sample(
        glob.glob('examples/non-vehicles/GTI/*') + glob.glob('examples/non-vehicles/Extras/*'),
        500) + glob.glob('examples/non-vehicles/my/*')))
cars = extract_features(load_images(random.sample(
        glob.glob('examples/vehicles/GTI_Far/*') 
        + glob.glob('examples/vehicles/GTI_Right/*')
        + glob.glob('examples/vehicles/KITTI_extracted/*'),
        1000)))

print("Total {} cars and {} non-cars".format(len(cars), len(not_cars)))

X = np.vstack((cars, not_cars))
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(cars)), np.zeros(len(not_cars))))

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3)

#%%
svc = LinearSVC(C=0.01, class_weight='balanced')
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

#%%
def analyze(path, wrong_predicate):
    wrong = []
    files2 = glob.glob(path)
    
    if len(files2) > 300:
        files2 = random.sample(files2, 300)

    features = X_scaler.transform(extract_features(load_images(files2)))
    predicted = svc.predict(features)
    wrong, = np.where(wrong_predicate(predicted))

    print("Accuracy: {}".format(1 - len(wrong) / len(files2)))        
    print("Wrong examples:")
    for w in islice(wrong, 1):
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(Image.open(files2[w]))
    plt.show()

print('Cars 1:')
analyze('car_images/*', lambda v: v == 0)
print('Cars 2:')
analyze('examples/vehicles/*/*', lambda v: v == 0)
print('Non-cars:')
analyze('examples/non-vehicles/Extras/*', lambda v: v != 0)


#%%
car_index = 0
heat = np.zeros(img.shape[0:3], dtype=np.int16)

def slide_window(img, heat):
    global car_index
    #heat = np.maximum(heat - 1, 0)
    heat = np.zeros(img.shape[0:3], dtype=np.int16)
    wss = [
            [64, 400, 600],
           # [96, 400, 500],
            [128, 380, 500],
            #[196, 400, 500],
           ]
    for params in wss:
        window_size = params[0]
        top = params[1]
        bottom = params[2]
        
        #heat = np.maximum(heat - 2, 0)
        h, w = img.shape[:2]
        xstep = ystep = window_size // 4
        for y in range(top, bottom, ystep):
            for x in range(0, w - window_size + 1, xstep):
                y1 = y
                x1 = x
                y2 = y + window_size
                x2 = x + window_size
                window = img[y1:y2, x1:x2]
                features = X_scaler.transform([extract_image_features(window)])
                #cv2.rectangle(heat, (x1, y1), (x2, y2), (200, 100, 255), 4)
                if svc.predict(features) == 1:
                    heat[y1:y2, x1:x2] += 1
                    #cv2.rectangle(heat, (x1, y1), (x2, y2), (100, 255, 150), 4)
                    car_index += 1
                    #scipy.misc.imsave("maybe_car {}.png".format(car_index), window)
                    #cv2.rectangle(img, (x1, y1), (x2, y2), (100, 255, 150), 4)
    
    heat[heat < 3] = 0
    labels = label(heat)
    
    for car_number in range(1, labels[1] + 1):
        car = (labels[0] == car_number).nonzero()
        cary = np.array(car[0])
        carx = np.array(car[1])
        bbox = ((np.min(carx), np.min(cary)), (np.max(carx), np.max(cary)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img
    
#%%
            
for i in glob.glob('seq2/*'):
#for i in glob.glob('test_images/*'):
    img = undistort(np.asarray(Image.open(i)))
    heat = slide_window(img, heat)
    plt.figure()
    plt.imshow(heat)
    plt.show()
    
#%%
from moviepy.editor import VideoFileClip

def process_video(file_name):
    process_video.heat = np.zeros(img.shape[0:3], dtype=np.int16)

    def process_frame(image):
        try:
            result = slide_window(image, process_video.heat)
        except Exception as ex:
            print("failed {}".format(ex))
            result = image

        return result

    clip = VideoFileClip(file_name)
    processed_clip = clip.fl_image(process_frame)
    processed_clip.write_videofile(file_name.replace('.', '-output.'), audio=False)
    
process_video("project_video.mp4")
    

#%%
for i in glob.glob('test_images/*'):
    img = np.asarray(Image.open(i))
    plt.figure()
    plt.imshow(img)
    plt.show()
    features, visual = extract_hog_features(img, visualize=True)
    for v in visual:
        plt.figure()
        plt.imshow(v)
        plt.show()


#%%
img = np.asarray(Image.open('test_images/test1.jpg'))
white_car = img[390:520,1040:1300]
black_car = img[400:500, 800:960]
#white_car = img[300:600, 950:1500]
#black_car = img[300:600, 700:1060]

def show_hog(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    features, visual = extract_hog_features(img, visualize=True)
    for v in visual:
        plt.figure()
        plt.imshow(v, cmap='gray')
        plt.show()


show_hog(white_car)
show_hog(black_car)

#%%

fig, axis = plt.subplots(ncols=2, nrows=11, figsize=(15, 30))

def show(img, ax):
    g = img[:,:,1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ax[0].imshow(img)
    
    h = hsv[:,:,0]
    ax[1].imshow(h, cmap='gray')
    ax[2].imshow(hsv[:,:,1], cmap='gray')
    ax[3].imshow(hsv[:,:,2], cmap='gray')

    ax[4].imshow(yuv[:,:,0], cmap='gray')
    ax[5].imshow(yuv[:,:,1], cmap='gray')
    ax[6].imshow(yuv[:,:,2], cmap='gray')
    
    red = np.copy(img)
    red[(2 < h) & (h < 175) | (g > 100)] = [0, 0, 0]
    ax[7].imshow(red)

    ax[8].imshow(ycrcb[:,:,0], cmap='gray')
    ax[9].imshow(ycrcb[:,:,1], cmap='gray')
    ax[10].imshow(ycrcb[:,:,2], cmap='gray')
    
show(white_car, axis[:,0])
show(black_car, axis[:,1])

    

