import torch
import numpy as np
import copy

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores""" 

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset() 

    def update(self, preds, labels): 
        """
        preds: "numpy array" or list of numpy array
        labels: "numpy array" or list of numpy array

        """ 
        def evaluate_worker(self, pred, label): 
            correct, labeled = batch_pix_accuracy(pred, label) 
            ignore_index = 0
            inter, union = batch_intersection_union(pred, label, self.nclass)
            torch.cuda.synchronize() 
            self.total_correct += correct.item()
            self.total_label += labeled.item()
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds,labels):
                evaluate_worker(self, pred_label)
        
    def get(self, return_category_iou = False):
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        if return_category_iou:
            return pixAcc, mIoU, IoU.cpu().numpy()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0)
    pixel_correct = torch.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass): 
    mini = 1 
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1 
    target = target.float() + 1

    predict = predict.float() * (target > 0).float() 
    intersection = predict * (predict == target).float() 

    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi) 
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area" 
    return area_inter.float(), area_union.float() 

def pixelAccuracy(imPred, imLab):
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled) 

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = imPred * (imPred == imLab) 
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union: 
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union) 

                                                                                                 
def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
   
