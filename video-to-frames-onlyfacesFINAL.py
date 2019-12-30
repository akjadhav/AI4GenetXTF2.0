import time
import os
import cv2

count = 0
image_number = 0

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.

    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.

    Returns:
        None
    """

    global count
    global image_number

    try:
        os.mkdir(output_loc)
    except OSError:
        pass

    time_start = time.time()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    print("Converting video..")

    # Start converting the video
    frame_number_taken = 0
    frames_saved = 0

    while cap.isOpened():
        ret, frame = cap.read()

        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = (cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if count % 25 == 0:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
            image_number += 1

            for x, y, w, h in faces:

                if int(.25 * h) > y:
                    frame_y_1 = y
                else:
                    frame_y_1 = y - int(.25 * h)

                if int(.18*h) > frame_height - (y + h):
                    frame_y_2 = frame_height - (y + h)
                else:
                    frame_y_2 = y + h + int((.18*h))

                if (frame_y_2-frame_y_1-w)/2 > x:
                    frame_x_1 = x
                else:
                    frame_x_1 = x - (frame_y_2-frame_y_1-w)/2

                if (frame_y_2-frame_y_1-w)/2 > frame_width - (x + w):
                    frame_x_2 = frame_width - (x + w)
                else:
                    frame_x_2 = x + w + (frame_y_2-frame_y_1-w)/2

                new_frame = frame[int(frame_y_1):int(frame_y_2), int(frame_x_1):int(frame_x_2)]

                try:
                    new_frame = cv2.resize(new_frame, (350, 350))
                    cv2.imwrite(output_loc + "/%#010d.jpg" % (image_number+3000000), new_frame)
                except:
                    print('Exception in line 75')

            frame_number_taken += 50
            frames_saved += 1

        count = count + 1

        if (frame_number_taken > (video_length-1)):
            time_end = time.time()
            cap.release()
            print('Done extracting frames.\n' + str(frames_saved) + ' frames saved\n\n')
            #print('It took ' + str(time_end-time_start) + ' seconds for conversion.')
            break


video_paths_location = r'Z:\Share\DownSyndrome\video_paths.txt'
output_folder = r'Z:\Share\DownSyndrome\videophotos'

with open(video_paths_location, 'r') as file:
    paths = file.readlines()

    for path in paths:
        path = path.rstrip('\n')
        path = path.strip(' ')
        video_to_frames(path, output_folder)