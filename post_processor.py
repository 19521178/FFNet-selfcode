import numpy as np

class PostProcessor(object):
    def __init__(self):
        pass
    def forward(self, summary):
        pass

class NoneExtend(PostProcessor):
    def __init__(self):
        pass
        
    def forward(self, summary):
        return summary
    
class ExtendAll(PostProcessor):
    def __init__(self, window_size, step_select):
        self.window_size = window_size
        self.step_select = step_select
        
    def forward(self, summary):
        temp_summary = summary.copy()
        summary_np = np.array(summary)
        indices_summ = np.where(summary_np==1)
        for index_gt in indices_summ[0]:
            for step in range(-self.window_size * self.step_select, self.window_size * self.step_select + 1, self.step_select):
                try:
                    temp_summary[index_gt + step] = 1
                except:
                    continue
        return temp_summary
    
class ExtendBoth(PostProcessor):
    def __init__(self, window_size, step_select, window_size_skip, step_select_skip, threshold):
        self.window_size = window_size
        self.step_select = step_select
        self.window_size_skip = window_size_skip
        self.step_select_skip = step_select_skip
        self.threshold = threshold
        
    def forward(self, summary):
        temp_summary = summary.copy()
        summary_np = np.array(summary)
        indices_summ = np.where(summary_np==1)
        if self.step_select_skip <= 0:
            self.step_select_skip = int(self.threshold/(self.window_size_skip*2+1))
        for index_indices_summ in range(len(indices_summ)):
            try:
                if indices_summ[index_indices_summ] - indices_summ[index_indices_summ-1] < self.threshold:  # skip action
                    if self.window_size_skip<0:
                        temp_summary[indices_summ[index_indices_summ]] = 0
                    else:
                        for step in range(-self.window_size_skip * self.step_select_skip, self.window_size_skip * self.step_select_skip+1, self.step_select_skip):
                            try:
                                temp_summary[indices_summ[index_indices_summ] + step] = 1
                            except:
                                continue
                        continue
            except:
                pass
            finally:
                for step in range(-self.window_size, self.window_size+1, 1):  # focus action
                    try:
                        temp_summary[indices_summ[index_indices_summ] + step] = 1
                    except:
                        continue
            
        return temp_summary
    
class ExtendAllBackward(PostProcessor):
    def __init__(self, window_size, window_size_backward, threshold):
        self.window_size = window_size
        self.window_size_backward = window_size_backward
        self.threshold = threshold
        
    def forward(self, summary):
        temp_summary = summary.copy()
        summary_np = np.array(summary)
        indices_summ = np.where(summary_np==1)
        for index_indices_summ in range(len(indices_summ)):
            action = indices_summ[index_indices_summ] - indices_summ[index_indices_summ-1]
            step_select_backward = int(action/(self.window_size_backward+1))
            try:
                # backward all action
                if self.window_size_backward<0:
                    temp_summary[indices_summ[index_indices_summ]] = 0
                else:
                    for step in range(-self.window_size_skip * step_select_backward, 0, step_select_backward):
                        temp_summary[indices_summ[index_indices_summ] + step] = 1
            except:
                pass
            finally:
                # focus action
                if action < self.threshold:
                    for step in range(-self.window_size, self.window_size+1, 1):
                        try:
                            temp_summary[indices_summ[index_indices_summ] + step] = 1
                        except:
                            continue
            
        return temp_summary
    
