# -*- coding: utf-8 -*-
"""
process the labled dataset - folder with folders.
choose and align only frontal faces. Then copy resized files to the dst folder
@author: IURII
"""


import cv2
import glob
import os
from shutil import copyfile



def main():
    #WITHOUT LAST "/" symbol
    src_folder = "E:/DataZoo/archive/192_192/total/vggface2_frontal"
    dst_folder = "E:/DataZoo/archive/192_192/total/images"

    full_counter = 0
    
    #create subdirs in aligned folder
    for subdir, dirs, files in os.walk(src_folder):
        #print(files)
        for dirct in dirs:
            
            #run within each subfolder
            files = [f for f in glob.glob(src_folder+"/"+dirct+"/" + "*")]
    
            for i, file in enumerate(files):  
                
                full_counter +=1
                #print(file)
                file = file.replace(os.sep, '/')
                     
                #defining filename and brisque filename
                new_filename = str(full_counter)+"_2_"+file.replace(src_folder+"/"+dirct+'/','')
                new_filename =  dst_folder+'/'+new_filename
  

                print(new_filename)
                copyfile(file, new_filename)
                    


    
    #Final results
    print(' all images - ', full_counter)

if __name__ == '__main__':
    main()