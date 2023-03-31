import numpy as np
import scipy.io as sio 
import os
from dataset import Dataset, TVSum
from post_processor import ExtendAll, NoneExtend


class Evaluator(object):
    def __init__(self):
        pass
    
    def load_dataset(self, dataset:Dataset):
        self.dataset = dataset
    
    def load_label(self, name, gt, predict):
        self.video_name = name
        self.y_hat = gt
        self.y = predict
        
    def reset_score_epoch():
        raise NotImplementedError()
    
    def get_score_vid(self, post_processor):
        raise NotImplementedError()
    
    def get_score_epoch(self):
        raise NotImplementedError()
    
    def get_score_epoch_from_summary_files(self, post_processor, summ_path):
        raise NotImplementedError()
    
class FScore(Evaluator):
    def __init__(self, precision_weight = 1, recall_weight = 1):
        self.recall_weight = recall_weight / precision_weight
        
    def reset_score_epoch(self):
        self.precisions = []
        self.recalls = []
        self.fscores = []
    
    def fscore_func(self, precision, recall):
        return (1 + self.recall_weight**2) * precision * recall / (self.recall_weight**2 * precision + recall)
    
    def get_score_vid(self, post_processor):
        summary = post_processor.forward(self.y)
        y_np = np.array(summary)
        y_hat_np = np.array(self.y_hat)
        
        true_pos = y_np * y_hat_np
        
        true_pos_sum = np.sum(true_pos)
        gt_sum = np.sum(y_hat_np)
        predict_sum = np.sum(y_np)
        
        recall = float(true_pos_sum)/float(gt_sum)
        precision = float(true_pos_sum)/float(predict_sum)  
        fscore = self.fscore_func(precision, recall)
        
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.fscores.append(fscore)
        return precision, recall, fscore
    
    def get_score_epoch(self):
        val_precision = np.mean(self.precisions)
        val_recall = np.mean(self.recalls)
        val_fscore = self.fscore_func(val_precision, val_recall)
        return val_precision, val_recall, val_fscore
    
    def get_score_epoch_from_summary_files(self, post_processor, summ_path):
        self.reset_score_epoch()
        for video_name in self.dataset.test_name:
            summary = sio.loadmat(os.path.join(summ_path, 'sum_' + video_name + '.mat'))['summary'][0]
            gt = self.dataset.get_video_gt(video_name)
            self.load_label(video_name, gt, summary)
            self.get_score_vid(post_processor)
        return self.get_score_epoch()
    
    @staticmethod
    def score_tostring(score):
        str_output = "%.2f" % score[0] + ' '\
                    +"%.2f" % score[1] + ' '\
                    +"%.2f" % score[2]
        return str_output

class Coverage(Evaluator):
    def __init__(self, hn_space):
        self.hn_space = hn_space
        try:
            self.num_hn = len(hn_space)
        except:
            self.num_hn = hn_space.shape[0]
            
    def count_tp_segment_per_hn_per_vid(self, summary, segments_gt):
        scores_hn = np.zeros(self.num_hn)
        tp_per_segment_gt = [sum(summary[start:end+1]) for start, end in segments_gt]
        for i in range(self.num_hn):
            scores_hn[i] += sum([int(tp>=self.hn_space[i]) for tp in tp_per_segment_gt])
        return scores_hn
    
    def reset_score_epoch(self):
        self.score_per_hn_epoch = np.zeros(self.num_hn)
        self.num_segment_gt_epoch = 0
    
    def get_score_vid(self, post_processor):
        import_segments = self.dataset.import_segment[self.video_name]
        num_segment_gt = len(import_segments)
        summary = post_processor.forward(self.y)
        score_per_hn = self.count_tp_segment_per_hn_per_vid(summary, import_segments)
        self.score_per_hn_epoch += score_per_hn
        self.num_segment_gt_epoch += num_segment_gt
        score_per_hn = score_per_hn / num_segment_gt
        return score_per_hn
        
    def get_score_epoch(self):
        return self.score_per_hn_epoch / self.num_segment_gt_epoch
    
    def get_score_epoch_from_summary_files(self, post_processor, summ_path):
        self.reset_score_epoch()
        for video_name in self.dataset.test_name:
            import_segments = self.dataset.import_segment[video_name] 
            summary = sio.loadmat(os.path.join(summ_path, 'sum_' + video_name + '.mat'))['summary'][0]
            summary = post_processor.forward(summary)
            score_per_hn = self.count_tp_segment_per_hn_per_vid(summary, import_segments)
            self.score_per_hn_epoch += score_per_hn
            self.num_segment_gt_epoch += len(import_segments)  
        return self.get_score_epoch()
    
    def score_tostring(self, scores):
        str_output = ''
        for hn, score in zip(self.hn_space, scores):
            str_output += str(hn) + ' ' + "%.2f" % score + '\n'
        return str_output

if __name__ == '__main__':
    # tvsum_dataset = TVSum(
    #     video_path='data/TVSum/video/',
    #     feat_path='data/TVSum/feat/',
    #     gt_path='data/TVSum/gt/'
    # )
    # test_name = ['J0nA4VgnoCo','vdmoEJ5YbrQ','0tmA_C6XwfM','Yi4Ij2NM7U4','XkqCExn6_Us','z_6gVvQb2d0','xmEERLqJ2kU','EE-bNr36nyA','eQu1rNs0an0','kLxoNp-UchI']
    # tvsum_dataset.split_train_test(test_name)
    
    none_post = NoneExtend()
    
    # summ_path = 'output/TVSum/'
    # f1_evaluator = FScore()
    # f1_evaluator.load_dataset(tvsum_dataset)
    # scores = f1_evaluator.get_score_epoch_from_summary_files(none_post, summ_path=summ_path)
    # print(f1_evaluator.score_tostring(scores))
    
    tvsum_dataset = TVSum(
        video_path='data/TVSum/video/',
        feat_path='data/TVSum/feat/',
        gt_path='data/TVSum/gt/'
    )
    test_name = ['J0nA4VgnoCo','vdmoEJ5YbrQ','0tmA_C6XwfM','Yi4Ij2NM7U4','XkqCExn6_Us','z_6gVvQb2d0','xmEERLqJ2kU','EE-bNr36nyA','eQu1rNs0an0','kLxoNp-UchI']
    tvsum_dataset.split_train_test(test_name)
    tvsum_dataset.load_gt_segments()
    
    w2_post = ExtendAll(window_size=2, step_select=1)
    
    summ_path = 'output/TVSum/'
    cover_evaluator = Coverage(np.arange(20) + 1)
    cover_evaluator.load_dataset(tvsum_dataset)
    scores = cover_evaluator.get_score_epoch_from_summary_files(none_post, summ_path)
    print(cover_evaluator.score_tostring(scores))
    
    
            