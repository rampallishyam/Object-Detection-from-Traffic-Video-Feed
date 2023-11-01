import cv2

__all__ = ['get_inputs']

class CorridorAnnotation:
    def __init__(self, frame):
        self.image = frame
        self.lines = []
        self.current_line = []

        cv2.namedWindow('Draw Corridor')
        cv2.setMouseCallback('Draw Corridor', self.click_event_line)

    def drawLine(self, line_points):
        if len(line_points) >= 2:
            start_point = line_points[0]
            end_point = line_points[1]
            color = (0, 255, 255)
            thickness = 2
            self.image = cv2.line(self.image, start_point, end_point, color, thickness)
        return self.image

    def click_event_line(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_line.append((x, y))
            if len(self.current_line) >= 2:
                self.lines.append(self.current_line)
                self.current_line = []

    def annotate_line(self):
        while True:
            for line in self.lines:
                self.image = self.drawLine(line)

            cv2.imshow('Draw Corridor', self.image)

            key = cv2.waitKey(1)
            if key == 27:  # Press Esc key to exit
                break

        cv2.destroyAllWindows()
        return self.lines

class ConversionPointAnnotations:
    def __init__(self, frame):
        self.image = frame
        self.points = []

        cv2.namedWindow('Input Conversion Points')
        cv2.setMouseCallback('Input Conversion Points', self.click_event_point)

    def drawPoint(self, point):
        # Set the point specs
        color = (0, 255, 255) 
        radius = 5

        # Draw the point on the image
        self.image = cv2.circle(self.image, (point[0],point[1]), radius, color, -1)

        return self.image

    def click_event_point(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) == 4:
                cv2.setMouseCallback('Input Conversion Points', lambda *args: None)  # Disable mouse callback
                cv2.destroyAllWindows()  # Close the window



    def annotate_point(self):
        while True and len(self.points)!=4:
            for point in self.points:
                self.image = self.drawPoint(point)

            cv2.imshow('Input Conversion Points', self.image)

            key = cv2.waitKey(1)
            if key == 27:  # Press Esc key to exit
                break

        cv2.destroyAllWindows()
        return self.points

def display_text(frame,frame_w,text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color_text = (255, 255, 255)  # BGR color format
    thickness = 2
    
    color_box = (0, 0, 0)  # BGR color format
    frame = cv2.rectangle(frame, (0,0), (frame_w, 60), color_box, -1)

    # Put the text on the image
    frame = cv2.putText(frame, text, (50, 50), font, font_scale, color_text, thickness)
    return frame

def get_inputs(video_path,LineAnnotation=CorridorAnnotation,PointAnnotaions=ConversionPointAnnotations,print_text_func=display_text):
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        exit()
    # Read the first frame of the video
    ret, frame = cap.read()
    width_frame = frame.shape[1]
    cap.release()
    # Check if a frame was successfully read
    if not ret:
        print("Error reading frame")
        exit()

    #annotate corridors
    print_text_func(frame,width_frame,"CLICK ON START AND END POINTS TO ADD CORRIDORS. PRESS ESC TO END") 
    annotation_line = LineAnnotation(frame)
    lines = annotation_line.annotate_line()

    #annotate Conversion Points
    print_text_func(frame,width_frame,"CLICK ON 4 CONVERSION POINTS (such that P1 & P2 gives Height and P2 & P3 give Widht )") 
    annotation_point = PointAnnotaions(frame)
    points = annotation_point.annotate_point()

    cv2.waitKey(0)
    # Release the video capture object and close the window
    cv2.destroyAllWindows()

    Real_Dim = []
    Real_Dim.append(int(input("Enter Real Height between P1 & P2 in meters \n")))
    Real_Dim.append(int(input("Enter Real Widht between P2 & P3 in meters \n")))
    

    return lines,points,Real_Dim



