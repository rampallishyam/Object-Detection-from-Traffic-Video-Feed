import argparse
import os
from typing import List,Dict
import numpy as np
import cv2
# from tqdm import tqdm
from ultralytics import YOLO
import torch
import csv
import pandas


from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from utils.detections2boxes import detections2boxes
from utils.get_tracks import get_distance,get_conv_mat
from utils.match_detections_with_tracks import match_detections_with_tracks
from utils.get_inputs import get_inputs


def main():
    #setting up main function call arguments
    parser = argparse.ArgumentParser(description='Run Model on Source Video')
    parser.add_argument('--model_yolo', default=MODEL, help='path for the Yolo model')
    parser.add_argument('--source_video', default=SOURCE_VIDEO_PATH, help='path of the input video')
    parser.add_argument('--interval_time', default=interval_time, help='time interval between logs in seconds')
    parser.add_argument('--class_ids', default=CLASS_ID,nargs='+', help='class_ids of interest') 
    args = parser.parse_args()
    args.interval_time = int(args.interval_time)
    lines,Pnts,real_dim = get_inputs(args.source_video)


    #creating RESULTS_DIR
    video_name = args.source_video.split('/')[-1]
    RESULTS_PATH = f"{HOME}/test/results/{video_name.split('.')[0]}_results"
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
        RUN_PATH = RESULTS_PATH + "/run1"
        os.mkdir(RUN_PATH)
        os.mkdir(RUN_PATH+"/corridor_counts")
        TRACK_PATH = RUN_PATH + "/tracks"
        os.mkdir(TRACK_PATH)

    else:
        runs_dir = os.listdir(RESULTS_PATH)
        tot_runs = sum(os.path.isdir(os.path.join(RESULTS_PATH, runs)) for runs in runs_dir)
        RUN_PATH = RESULTS_PATH + f"/run{tot_runs+1}"
        os.mkdir(RUN_PATH)
        os.mkdir(RUN_PATH+"/corridor_counts")
        TRACK_PATH = RUN_PATH + "/tracks"
        os.mkdir(TRACK_PATH)

    #Yolo Model 
    model = YOLO(args.model_yolo)
    model.fuse()
    CLASS_NAMES_DICT = model.model.names


    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())

    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(args.source_video)
    
    # create LineCounter instances
    line_counters = []
    for line in lines:
        if len(line)==2:
            line_counter = LineCounter(start=Point(line[0][0], line[0][1]), end=Point(line[1][0], line[1][1]),all_vehicle_class=args.class_ids)
            line_counters.append(line_counter)

    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # open target video file
    with VideoSink(RUN_PATH +"/result_video.mp4", video_info) as sink:
        video = cv2.VideoCapture(args.source_video)
        if not video.isOpened():
            raise Exception(f"Could not open video at {args.source_video}")

        #conversion matrix to convert Pixels to Real-World Coordinates
        conv_mat = get_conv_mat(Pnts,real_dim)
        #stores turning proportions
        vehicle_turns: Dict[str, list] = {}

        #Creating Vehicle Count .csv
        vehicle_count_header = ['Time Interval (seconds)',] + [CLASS_NAMES_DICT[id] for id in args.class_ids] + ['Total (cumulative)',]
        flows_type = ['inflow','outflow']
        for i in range(len(line_counters)):
            for j,flw_typ in enumerate(flows_type): 
                with open(RUN_PATH + f"/corridor_counts/vehicle_counts_corridor{i+1}_{flw_typ}.csv", 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(vehicle_count_header)

         
        # loop over video frames
        for i in range((video_info.total_frames//(video_info.fps*args.interval_time))+1):

            #To get the no.of frames to iter over
            if i == (video_info.total_frames//(video_info.fps*args.interval_time)):
                iter_frames = (video_info.total_frames%(video_info.fps*args.interval_time))
            else:
                iter_frames = (video_info.fps*args.interval_time)

            #Appending Vehicle Count .csv
            for j in range(len(line_counters)):
                total_counts_this = line_counters[j].get_data()
                for o,flw_typ in enumerate(flows_type):
                    with open(RUN_PATH + f"/corridor_counts/vehicle_counts_corridor{j+1}_{flw_typ}.csv", 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([i*args.interval_time,] + [cnt[o] for cnt in total_counts_this.values()])

            pos_real = {} #intializing positions_dict

            #reading frames and itering over all the frames
            success, frame = video.read()
            frame_count=0

            while success and frame_count<iter_frames:

                for l in range(4):
                    cv2.circle(frame, (Pnts[l][0],Pnts[l][1]), 5, (0, 255, 255), -1)

                # model prediction on single frame and conversion to supervision Detections
                results = model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )

                # filtering out detections with unwanted classes
                mask = np.array([class_id in args.class_ids for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)

                # tracking detections
                tracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )

                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)

                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True) 
                
                # format custom labels
                labels = [
                    f"#{tracker_id}"
                    for _, _, _, tracker_id
                    in detections
                ]
                # labels = [
                #     f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                #     for _, confidence, class_id, tracker_id
                #     in detections
                # ]

                #track real-world positions of vehicles w.r.t time for specific tracker_id
                get_distance(conv_mat, detections, pos_real,frame_count)
                
                # updating line counter
                for k in range(len(line_counters)):
                    line_counters[k].update(detections=detections,corridor_id = k+1,turning_prop=vehicle_turns)
                
                # annotate and display frame
                frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                for k in range(len(line_counters)):
                    line_annotator.annotate(frame=frame, line_counter=line_counters[k])
                sink.write_frame(frame)
                
                
                success, frame = video.read()
                frame_count=frame_count+1
            

            for track_id in list(pos_real.keys()):

                #calculating distance travelled and speed
                distance = np.sqrt((pos_real[track_id][-1][0] - pos_real[track_id][0][0])**2 + (pos_real[track_id][-1][1] - pos_real[track_id][0][1])**2 )
                time_interval = (pos_real[track_id][-1][2] - pos_real[track_id][0][2])/video_info.fps
                speed = 3.6*abs(distance)/time_interval
                
                #logging trajectories and speeds
                if os.path.exists(TRACK_PATH + f"/{track_id}.csv"):
                    with open(TRACK_PATH + f"/{track_id}.csv", 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([np.round(i*args.interval_time + pos_real[track_id][0][2]/video_info.fps),np.round(i*args.interval_time + pos_real[track_id][-1][2]/video_info.fps),pos_real[track_id][0][0],pos_real[track_id][0][1],distance,speed])
                
                else:
                    with open(TRACK_PATH + f"/{track_id}.csv", 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Time Start (seconds)','Time End (seconds)', 'Real_X (meters)', 'Real_Y (meters)', 'Distance Travelled (meters)', 'Speed (km/hr)'])
                        writer.writerow([np.round(i*args.interval_time + pos_real[track_id][0][2]/video_info.fps),np.round(i*args.interval_time + pos_real[track_id][-1][2]/video_info.fps),pos_real[track_id][0][0],pos_real[track_id][0][1],distance,speed])

        video.release()
    
    OD_matrix:  Dict[tuple, int] = {}
    for id in list(vehicle_turns.keys()):
        if len(vehicle_turns[id])>=2:
            try:
                OD_matrix[tuple(vehicle_turns[id])] += 1
            except:
                OD_matrix.setdefault(tuple(vehicle_turns[id]), 1)
    
    with open(RUN_PATH + f"/OD_matrix.csv", 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["v-- O , D -->",] + ["corridor_"+str(i) for i in range(1,1+len(line_counters))])
                    for i in range(1,1+len(line_counters)):
                        temp=np.zeros(len(line_counters))
                        for od in OD_matrix.keys():
                            if od[0]==i:
                                temp[od[-1]-1] = OD_matrix[od]
                        writer.writerow(["corridor_"+str(i),] + list(temp))

    


if __name__ == '__main__':
    #cuda avaiblility check
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device," ",torch.version.cuda)

    HOME = os.getcwd() 

    #model initialization
    MODEL = f"{HOME}/models/all_weights/idd_dataset.pt"     
    
    CLASS_ID = [0,1,2,3,4] # class_ids of Indian_Traffic_model - bike, auto, car, bus and truck
    # CLASS_ID = [2, 3, 5, 7] # class_ids of orginal yolov8 - car, motorcycle, bus and truck

    #input path
    SOURCE_VIDEO_PATH = f"{HOME}/test/test_videos/video1.mp4" 
    #time interval between logs ( in seconds )
    interval_time = 2

    # Real_Dim = [25,30]  # height , width in  meters
    main()


