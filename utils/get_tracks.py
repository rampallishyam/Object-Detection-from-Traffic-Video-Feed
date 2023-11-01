import numpy as np
__all__ = ['get_distance','get_conv_mat']

def get_conv_mat(pnts,real_dim):
    
    U=[0,0,real_dim[1],real_dim[1]]
    V=[0,real_dim[0],real_dim[0],0]

    B = np.array(U+V)
    # print(f"B = {B}")
    
    A = np.zeros((8,8))
    for i in range(8):
        if i<4:
            A[i] = np.array([pnts[i][0] , pnts[i][1], 1, 0, 0, 0, -1*pnts[i][0]*B[i], -1*pnts[i][1]*B[i] ])
        else: 
            A[i] = np.array([0, 0, 0, pnts[i-4][0] , pnts[i-4][1], 1, -1*pnts[i-4][0]*B[i], -1*pnts[i-4][1]*B[i] ])
    # print(f"A = {A}")

    return np.linalg.solve(A,B)
    
    # try:
    #     return np.linalg.solve(A,B)

    # except:

    #     return np.zeros(8)



def get_distance(conv_mat,detections,postions,frame_no):
    
    for xyxy, _, _, tracker_id in detections:
        # handle detections with no tracker_id
        if tracker_id is None:
            continue
        
        # calculate vehicle_width_in_frame
        x1, y1, x2, y2 = xyxy

        #finding the center to represent vehicle with single point
        x_cm = np.mean([x1,x2])
        y_cm = np.mean([y1,y2])

        #formula to convert from pixel values to real_coordinates
        x_real = ( conv_mat[0]*x_cm + conv_mat[1]*y_cm + conv_mat[2] ) / ( conv_mat[-2]*x_cm + conv_mat[-1]*y_cm + 1 )
        y_real = ( conv_mat[3]*x_cm + conv_mat[4]*y_cm + conv_mat[5] ) / ( conv_mat[-2]*x_cm + conv_mat[-1]*y_cm + 1 )
        
        if tracker_id not in list(postions.keys()):
            postions[tracker_id] = [(x_real,y_real,frame_no),]

        else:
            postions[tracker_id].append((x_real,y_real,frame_no))









                





	
	
	