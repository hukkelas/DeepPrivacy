import torch
import math
from . import torch_utils
from torchvision.ops.boxes import nms


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]    
    # (cx,cy,w,h)->(x0,y0,x1,y1)
    return boxes


class Detect:
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, variance):
        self.variance = variance

    def forward(self, loc_data, conf_data, prior_data, confidence_threshold, nms_threshold):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors,  2).transpose(2, 1)

        final_ouput = []
        for i in range(num):

            default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            conf_scores = conf_preds[i, 1]
            # Very costly operation, conf_score shape: [172845]
            indices = (conf_scores >= confidence_threshold).nonzero().squeeze()
            decoded_boxes = decoded_boxes[indices]

            conf_scores = conf_scores[indices]
            if conf_scores.dim() == 0:
                final_ouput.append(torch.empty(0, 5))
            keep_idx = nms(decoded_boxes, conf_scores, nms_threshold)

            scores = conf_scores[keep_idx].view(1, -1, 1)
            boxes = decoded_boxes[keep_idx].view(1, -1, 4)
            output = torch.cat((scores, boxes), dim=-1)
            final_ouput.append(output)
        if num == 1:
            return final_ouput[0]
        final_ouput = torch.cat(final_ouput, dim=0)
        return final_ouput


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg , image_size, feature_maps):
        super(PriorBox, self).__init__()
        self.image_size = image_size
        self.feature_maps = feature_maps
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.min_sizes = cfg["min_sizes"]

        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []

        if len(self.min_sizes) == 5:
            self.feature_maps = self.feature_maps[1:]
            self.steps = self.steps[1:]
        if len(self.min_sizes) == 4:
            self.feature_maps = self.feature_maps[2:]
            self.steps = self.steps[2:]

        for k, f in enumerate(self.feature_maps):
            #for i, j in product(range(f), repeat=2):
            for i in range(f[0]):
              for j in range(f[1]):
                #f_k = self.image_size / self.steps[k]
                f_k_i = self.image_size[0] / self.steps[k]
                f_k_j = self.image_size[1] / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k_j
                cy = (i + 0.5) / f_k_i
                # aspect_ratio: 1
                # rel size: min_size
                s_k_i = self.min_sizes[k]/self.image_size[1]
                s_k_j = self.min_sizes[k]/self.image_size[0]
                # swordli@tencent
                if len(self.aspect_ratios[0]) == 0:
                    mean += [cx, cy, s_k_i, s_k_j]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                #s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                if len(self.max_sizes) == len(self.min_sizes):
                    s_k_prime_i = math.sqrt(s_k_i * (self.max_sizes[k]/self.image_size[1]))
                    s_k_prime_j = math.sqrt(s_k_j * (self.max_sizes[k]/self.image_size[0]))    
                    mean += [cx, cy, s_k_prime_i, s_k_prime_j]
                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    if len(self.max_sizes) == len(self.min_sizes):
                        mean += [cx, cy, s_k_prime_i/math.sqrt(ar), s_k_prime_j*math.sqrt(ar)]
                    mean += [cx, cy, s_k_i/math.sqrt(ar), s_k_j*math.sqrt(ar)]
                
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        output = torch_utils.to_cuda(output)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
