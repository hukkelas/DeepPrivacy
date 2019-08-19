import torch
import torch.nn.functional as F
# torch.backends.cudnn.bencmark = True

import cv2
import numpy as np

from src.detection.SFD_pytorch.net_s3fd import s3fd
from src.detection.SFD_pytorch.bbox import nms, decode
from apex import amp
from src.torch_utils import to_cuda
net = s3fd()
net.load_state_dict(torch.load('src/detection/SFD_pytorch/data/s3fd_convert.pth'))
if torch.cuda.is_available():
    net.cuda()
net.eval()
to_cuda(net)
#net = amp.initialize([net], opt_level="O1")[0]

def detect(net,img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = torch.from_numpy(img).float().cuda()
    with torch.no_grad():
        olist = net(img)

    bboxlist = []
    for i in range(len(olist)//2): 
        #print(olist[i*2])
        olist[i*2] = F.softmax(olist[i*2], dim=1)
    for i in range(len(olist)//2):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        _, _, FH, FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

def flip_detect(net,img):
    img = cv2.flip(img, 1)
    with torch.no_grad():
        b = detect(net,img)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist

def scale_detect(net,img,scale=2.0,facesize=None):
    img = cv2.resize(img,(0,0),fx=scale,fy=scale)
    with torch.no_grad():
        b = detect(net,img)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = b[:, 0]/scale
    bboxlist[:, 1] = b[:, 1]/scale
    bboxlist[:, 2] = b[:, 2]/scale
    bboxlist[:, 3] = b[:, 3]/scale
    bboxlist[:, 4] = b[:, 4]
    b = bboxlist
    if scale>1: index = np.where(np.minimum(b[:,2]-b[:,0]+1,b[:,3]-b[:,1]+1)<facesize)[0] # only detect small face
    else: index = np.where(np.maximum(b[:,2]-b[:,0]+1,b[:,3]-b[:,1]+1)>facesize)[0] # only detect large face
    bboxlist = b[index,:]
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

def detect_and_supress(img, score_threshold):
    with torch.no_grad():
        resize_ratio = 1.0
        if max(img.shape) > 1080:
            resize_ratio = 1080 / max(img.shape)
            img = cv2.resize(img, (0,0), fx=resize_ratio, fy=resize_ratio)
        bbox1 = detect(net, img)
        bbox2 = flip_detect(net, img)
        bbox3 = np.zeros((1, 5))
        if img.shape[0]*img.shape[1]*4 <= 3000*3000:
            bbox3 = scale_detect(net, img, scale=2, facesize=100)
        bbox4 = scale_detect(net, img, scale=0.5, facesize=100)
        bboxes = np.concatenate((bbox1, bbox2, bbox3, bbox4))
        keep = nms(bboxes, 0.3)[:750]
        bboxes = bboxes[keep]
        bboxes = bboxes[bboxes[:, 4] >= 0.5] # Remove small faces
        
        bboxes[:, :4] /= resize_ratio
        scores = bboxes[:, 4]
        bboxes = bboxes[scores > score_threshold, :]
        scores = bboxes[:, 4]
        sorted_ix = np.argsort(scores)[::-1]
        bboxes = bboxes[sorted_ix]

        bboxes = bboxes.astype("int")
        return bboxes[:, :4]

if __name__ == "__main__":
    img = cv2.imread("maxresdefault.jpg")
    detect_and_supress(img)

