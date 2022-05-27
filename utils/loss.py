# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou, xywh2xyxy, clip_coords
from utils.torch_utils import is_parallel
import math


COCO_IMG_W = 640
COCO_IMG_H = 480
COCO_SMALL_T = 32
COCO_MEDIUM_T = 96


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, var=None):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                if var:
                    vars = var[i][b, a, gj, gi]
                    
                    lbox_xy = -torch.log(self.Gaussian(tbox[i][..., :2], pbox[..., :2], vars[..., :2]) + 10**-9)/2.0
                    lbox_wh = -torch.log(self.Gaussian(tbox[i][..., 2:], pbox[..., 2:], vars[..., 2:]) + 10**-9)/2.0
                    # lbox += torch.unsqueeze(torch.mean(torch.cat((lbox_xy, lbox_wh), axis=1).sum(axis=1)), axis=0) * 0.01
                    lbox += torch.unsqueeze((torch.cat((lbox_xy, lbox_wh), axis=1).sum(axis=1)).sum(), axis=0)
                else:
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 0.0005
        # lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def Gaussian(self, y, mu, var):
        var = torch.sigmoid(var)
        
        eps = 0.3
        result = (y-mu)/var
        result = (result**2)/2*(-1)
        exp = torch.exp(result)
        result = exp/(math.sqrt(2*math.pi))/(var + eps)

        return result


class ComputeDomainLoss:
    # Compute domain losses
    def __init__(self, model):
        h = model.hyp  # hyperparameters

        # Define criteria
        BCE = nn.BCEWithLogitsLoss()
        self.BCE, self.hyp = BCE, h 

    def __call__(self, sp, tp):  # source predictions, target predictions
        device = sp[0].device

        losses = [torch.zeros(1, device=device) for _ in range(len(sp))]
        accuracies = [torch.zeros(1, device=device) for _ in range(len(sp))]
        targets = self.build_targets(sp, tp)  # targets

        # Losses and accuracies
        for i in range(len(sp)):
            losses[i] += self.BCE(torch.cat((sp[i], tp[i])), targets[i].to(device))
            accuracies[i] = self.compute_accuracies(torch.cat((sp[i], tp[i])), targets[i].to(device))

        bs = sp[0].shape[0]
        
        return sum(losses)*bs*0.05, torch.cat(losses).detach(), torch.cat(accuracies).detach()

    def build_targets(self, sp, tp):
        # Build targets for compute_domain_loss()
        t = []
        for i in range(len(sp)):
            t.append(torch.cat((torch.zeros(sp[i].shape), torch.ones(tp[i].shape))))
        return t

    def compute_accuracies(self, scores, ground_truth):
        # Compute accuracies for compute_domain_loss()
        predictions = (scores > 0.) # if > 0 it predicted source
        num_correct = (predictions == ground_truth).sum()
        num_samples = torch.prod(torch.tensor(predictions.shape))
        accuracy = float(num_correct)/float(num_samples)*100
        return torch.tensor([accuracy]).to(scores.device)


class ComputeAttentionLoss:
    # Compute attention losses
    def __init__(self, model):
        h = model.hyp  # hyperparameters
        self.device = next(model.parameters()).device  # get model device

        # Define criteria
        Dice = DiceLoss()

        self.Dice, self.hyp = Dice, h 

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        for k in 'na', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, attn_maps, sep_targets):  # objectness maps, targets
        lattn = [torch.zeros(1, device=self.device) for _ in range(len(attn_maps))]
        tattn = self.build_COCO_targets(attn_maps, sep_targets)  # targets

        # Losses
        for i, attn_map in enumerate(attn_maps):
            lattn[i] += self.Dice(attn_map, tattn[i])

        bs = attn_maps[0].shape[0]

        return sum(lattn)*bs*0.005, torch.cat(lattn).detach()

    def build_targets(self, attn_maps, sep_targets):
        tattns = [torch.zeros([0]).to(self.device) for _ in range(len(attn_maps))]

        for targets in sep_targets:
            targets = targets.to(self.device)
            # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
            na, nt = self.na, targets.shape[0]  # number of anchors, targets
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
            ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
            targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

            for i in range(self.nl):
                h, w = attn_maps[i].shape[1:]
                attn_mask = torch.zeros((h, w)).to(self.device)
                anchors = self.anchors[i]
                gain[2:6] = torch.tensor([[w, h, w, h]])  # xyxy gain

                # Match targets to anchors
                t = targets * gain
                if nt:
                    # Matches
                    r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                    j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                    # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                    t = t[j]  # filter
                    t = torch.unique(t[:, 2:6], dim=0)  # filter

                    if len(t):
                        # Define
                        tbox = xywh2xyxy(t)
                        clip_coords(tbox, (h, w))
                        for xyxy in tbox:
                            left = xyxy[0].round().int()
                            top = xyxy[1].round().int()
                            right = xyxy[2].round().int()
                            bottom = xyxy[3].round().int()
                            attn_mask[top:(bottom+1), left:(right+1)] = 1

                # Append
                tattns[i] = torch.cat((tattns[i], torch.unsqueeze(attn_mask, dim=0)), dim=0)

        return tattns

    def build_COCO_targets(self, attn_maps, sep_targets):
        tattns = [torch.zeros([0]).to(self.device) for _ in range(len(attn_maps))]

        for targets in sep_targets:
            targets = targets.to(self.device)
            # Build binary mask for compute_loss(), input targets(image,class,x,y,w,h)
            small_t = (COCO_SMALL_T / COCO_IMG_W) * (COCO_SMALL_T / COCO_IMG_H)
            medium_t = (COCO_MEDIUM_T / COCO_IMG_W) * (COCO_MEDIUM_T / COCO_IMG_H)
            
            small_mask = torch.zeros(targets.shape[0], device=targets.device, dtype=torch.bool)
            medium_mask = torch.zeros(targets.shape[0], device=targets.device, dtype=torch.bool)
            large_mask = torch.zeros(targets.shape[0], device=targets.device, dtype=torch.bool)

            # Define
            box_area = targets[:, 4] * targets[:, 5]
            small_mask = small_mask.add((box_area < small_t))
            medium_mask = medium_mask.add((box_area > small_t) & (box_area < medium_t))
            large_mask = large_mask.add((box_area > medium_t))
            masks = [small_mask, medium_mask, large_mask]
            
            nt = targets.shape[0]  # number of targets
            gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

            for i in range(self.nl):
                h, w = attn_maps[i].shape[1:]
                attn_mask = torch.zeros((h, w)).to(self.device)
                gain[2:6] = torch.tensor([[w, h, w, h]])  # xyxy gain

                # Match targets to anchors
                t = targets * gain
                if nt:
                    # Matches
                    t = t[masks[i]]
                else:
                    t = targets

                # Define
                tbox = xywh2xyxy(t[:, 2:6])
                clip_coords(tbox, (h, w))
                for xyxy in tbox:
                    left = xyxy[0].round().int()
                    top = xyxy[1].round().int()
                    right = xyxy[2].round().int()
                    bottom = xyxy[3].round().int()
                    attn_mask[top:(bottom+1), left:(right+1)] = 1

                # Append
                tattns[i] = torch.cat((tattns[i], torch.unsqueeze(attn_mask, dim=0)), dim=0)

        return tattns


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice