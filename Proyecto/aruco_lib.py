import numpy as np
import cv2

class aruco_lib:
    def __init__(self):
        self.imagen_final = None
        self.ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

    def create_aruco_marker(self, aruco_type, id):
        arucoDict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[aruco_type])
        #print("ArUCo type '{}' with ID '{}'".format(aruco_type, id))

        tag_size = 250
        self.imagen_final = np.zeros((tag_size, tag_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(arucoDict, id, tag_size, self.imagen_final, 1)
        
        return self.imagen_final

    def aruco_detection(self, img, aruco_type):
        arucoDict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        return self.aruco_display(corners, ids, rejected, img)

    def aruco_display(self, corners, ids, rejected, image):
        centers = np.zeros((4,2), dtype="int")

        if len(corners) > 0:
            ids = ids.flatten()
            
            for (markerCorner, markerID) in zip(corners, ids):
                
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)


                print("[Inference] ArUco marker ID: {}".format(markerID))
                #print("Coordenadas ->")
                #print("TopRight = x:"+str(topRight[0])+" y:"+str(topRight[1]))
                #print("bottomRight = x:"+str(bottomRight[0])+" y:"+str(bottomRight[1]))
                #print("bottomLeft = x:"+str(bottomLeft[0])+" y:"+str(bottomLeft[1]))
                #print("topLeft = x:"+str(topLeft[0])+" y:"+str(topLeft[1]))
                print("Center -> x:"+str(cX) + " y:" + str(cY))
                centers[markerID-1][0] = cX
                centers[markerID-1][1] = cY
        else:
            print("corners not found ")
        return image, centers


def get_img(aruco_type, id):
    res = aruco_lib()
    return res.create_aruco_marker(aruco_type, id)

def detect_aruco_markers(img, aruco_type):
    res = aruco_lib()
    return res.aruco_detection(img, aruco_type)
