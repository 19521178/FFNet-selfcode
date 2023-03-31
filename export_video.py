import cv2
import numpy as np 
import scipy.io as sio
from post_processor import NoneExtend, PostProcessor
import time

class Exporter(object):
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        
        self.init_writer()
        
    def load_label(self, summary):
        self.summary = summary
        
    def init_writer(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        
    def export(self, video_name, post_processor:PostProcessor = NoneExtend()):
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path + video_name + '.MP4')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(self.output_path + video_name + '.mp4', \
                        self.fourcc, fps, (width, height), isColor=True)
        
        summary = post_processor.forward(self.summary)
        success, frame = cap.read()
        index = 0
        while success:
            if summary[index]==1:
                out.write(frame)
            index += 1
            success, frame = cap.read()
            
        end_time = time.time()
        cap.release()
        out.release()
        
        percent_process = np.mean(self.summary)
        percent_selected = np.mean(summary)
        return percent_process, percent_selected, end_time - start_time
        
    def export_from_summary_files(self, summ_path, video_names = [], post_processor:PostProcessor = NoneExtend()):
        for video_name in video_names:
            print('Exporting '+ video_name, end=' | ')
            summary = sio.loadmat(summ_path + 'sum_' + video_name + '.mat')['summary'] #shape (1, n_frame)
            self.load_label(summary[0])
            
            percent_process, percent_selected, run_time = self.export(video_name, post_processor=post_processor)
            print('Process: ' + '%.2f' % percent_process +\
                ' | Selected: ' + '%.2f' % percent_selected + \
                ' | Runtime: ' + '%.0f' % run_time +\
                ' | Saved')
            
if __name__ == '__main__':
    video_path = 'data/TVSum/video/'
    summ_path = 'output/TVSum/summary/'
    save_path = 'output/TVSum/video/'
    test_name = ['J0nA4VgnoCo','vdmoEJ5YbrQ','0tmA_C6XwfM','Yi4Ij2NM7U4','XkqCExn6_Us','z_6gVvQb2d0','xmEERLqJ2kU','EE-bNr36nyA','eQu1rNs0an0','kLxoNp-UchI']

    
    post_processor = NoneExtend()
    
    exporter = Exporter(video_path, save_path)
    exporter.export_from_summary_files(summ_path=summ_path, video_names=test_name, post_processor=post_processor)
    
            
            
            
            
            