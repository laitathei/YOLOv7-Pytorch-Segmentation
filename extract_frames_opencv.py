from datetime import timedelta
import cv2
import numpy as np
import os

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 2
id = 0 # change it if do building some dataset with different video source
        # if source 1 have 55 images, you should change to id = 55
        # then source 2 have 66, you should change to id = 55+66

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_file):
    video_name = video_file.split(".")
    video_name = video_name[0]
    video_name = ''.join([i for i in video_name if not i.isdigit()])
    # filename, _ = os.path.splitext(video_file)
    # make a folder by the name of the video file
    if not os.path.isdir(video_name):
        os.mkdir(video_name)
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    count = 0
    internal_count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break

        if frame_duration >= closest_duration:
            cv2.imwrite(os.path.join(video_name, f"{video_name}{internal_count+id}.jpg"), frame) 
            print(os.path.join(video_name, f"{video_name}{internal_count+id}.jpg"))
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
                internal_count += 1
            except IndexError:
                pass
        # increment the frame count
        count += 1
    print("number of images: {}".format(internal_count+id))

if __name__ == "__main__":
    import sys
    video_file = sys.argv[1]
    main(video_file)

# python3 extract_frames_opencv.py zoo.mp4
# all jpg will store in dataset
# replace video_name with your input video name, if zoo.mp4 means zoo000000.mp4
