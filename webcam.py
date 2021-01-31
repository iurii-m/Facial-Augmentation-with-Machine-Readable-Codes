# -*- coding: utf-8 -*-
"""

Class for running webcam in thread.
Taken from http://devres.zoomquiet.top/data/20160323155130/index.html

"""

import cv2
from threading import Thread


class Webcam:
  
    def __init__(self, if_show = True):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]
        self.if_show = if_show
          
    # create thread for capturing images
    def start(self):
        
        webcam_thread = Thread(target=self._update_frame, args=())
        webcam_thread.start()
        return(webcam_thread)
  
    def _update_frame(self):
        while(True):
            self.current_frame = self.video_capture.read()[1]
            if self.if_show:
                cv2.imshow('Source Webcam Stream', self.current_frame)
                cv2.waitKey(1)
                  
    # get the current frame
    def get_current_frame(self):
        return self.current_frame
    
        