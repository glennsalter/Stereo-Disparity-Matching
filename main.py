import time
import math
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from pathlib import Path

args = {}
args["maxdisp"] = 192

# TODO: Modify directories for images and model
args["imagedir"] = "/Users/glennsalter/Documents/cv-hw2/images"
args["loadmodel"] = "/Users/glennsalter/Documents/cv-hw2/models/finetune_250.tar"

if not torch.cuda.is_available():
    raise ImportError("Need cuda")

if cv2.ximgproc is None:
    raise ImportError("Need to install opencv-contrib-python==4.7.0.72")

torch.cuda.manual_seed(1)

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def run_stereo_sgbm(leftimg, rightimg):
    # Load the left and right images and their corresponding keypoints and descriptors
    img_left = cv2.imread(leftimg, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(rightimg, cv2.IMREAD_GRAYSCALE)

    # Set the SGBM parameters
    win_size = 25
    min_disp = 0
    num_disp = 256
    uniqueness_ratio = 10
    speckle_window_size = 100
    speckle_range = 32
    disp_max_diff = 100

    # Create the StereoSGBM object and compute the disparity map
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=win_size,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=disp_max_diff,
        P1=8*3*win_size**2,
        P2=32*3*win_size**2
    )

    disparity = stereo.compute(img_left, img_right)
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_norm

def postprocess_inpaint(disparity_map):
    # Create a binary mask where 0 pixels indicate missing data
    mask = np.zeros_like(disparity_map)
    mask[disparity_map < 70] = 255

    # Apply Depth Map Inpainting algorithm to fill in missing values
    inpainting_method = cv2.INPAINT_NS
    inpainting_radius = 5
    inpaint = cv2.inpaint(disparity_map, mask, inpainting_radius, inpainting_method)
    return inpaint

def postprocess_mean_filter(disparity_map):
    mean_filter = cv2.blur(disparity_map, (5, 5))
    return mean_filter

def postprocess_wls_filter(disparity_map):
    # Create a binary mask where 0 pixels indicate missing data
    mask = np.zeros_like(disparity_map)
    mask[disparity_map < 70] = 255

    # Define the parameters for the WLS filter
    sigma_s = 60
    sigma_r = 0.25

    # Apply the WLS filter to fill in the missing regions
    wls_filter = cv2.ximgproc.guidedFilter(disparity_map, mask, radius=10, eps=1e-6, dDepth=-1)
    wls_filter = cv2.ximgproc.weightedMedianFilter(wls_filter, disparity_map, 5, sigma_s, sigma_r)
    return wls_filter

def postprocess_inpaint_wls(disparity_map):
    # Create a binary mask where 0 pixels indicate missing data
    mask = np.zeros_like(disparity_map)
    mask[disparity_map < 70] = 255

    # Apply Depth Map Inpainting algorithm to fill in missing values
    inpainting_method = cv2.INPAINT_NS
    inpainting_radius = 5
    inpaint = cv2.inpaint(disparity_map, mask, inpainting_radius, inpainting_method)

    mask = np.zeros_like(inpaint)
    mask[inpaint < 70] = 255

    # Define the parameters for the WLS filter
    sigma_s = 60
    sigma_r = 0.25

    # Apply the WLS filter to fill in the missing regions
    inpaint_wls = cv2.ximgproc.guidedFilter(inpaint, mask, radius=10, eps=1e-6, dDepth=-1)
    inpaint_wls = cv2.ximgproc.weightedMedianFilter(inpaint_wls, inpaint, 5, sigma_s, sigma_r)
    
    return inpaint_wls

def postprocess_inpaint_mean(disparity_map):
    # Create a binary mask where 0 pixels indicate missing data
    mask = np.zeros_like(disparity_map)
    mask[disparity_map < 70] = 255

    # Apply Depth Map Inpainting algorithm to fill in missing values
    inpainting_method = cv2.INPAINT_NS
    inpainting_radius = 5
    inpaint = cv2.inpaint(disparity_map, mask, inpainting_radius, inpainting_method)

    inpaint_mean = cv2.blur(inpaint, (5, 5))
    
    return inpaint_mean

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

def predict(model, imgL, imgR):
    model.eval()

    imgL = imgL.cuda()
    imgR = imgR.cuda()   

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp

def run_psmnet(model, leftimg, rightimg, outputimg):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(**normal_mean_var)])    

    imgL_o = Image.open(leftimg).convert('RGB')
    imgR_o = Image.open(rightimg).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 


    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    pred_disp = predict(model, imgL,imgR)

    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        img = pred_disp[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp

    img = (img*256).astype('uint16')
    img = Image.fromarray(img)
    img.save(outputimg)
    return cv2.imread(outputimg, cv2.IMREAD_GRAYSCALE)

def postprocess_fill_in_missing(source, dest):
    mask = dest < 70
    dest[mask] = source[mask]
    return dest

def postprocess_avg(disp1, disp2):
    avg_img = cv2.addWeighted(disp1, 0.5, disp2, 0.5, 0)
    return avg_img

if __name__=='__main__':
    model = PSMNet(args["maxdisp"])
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
    state_dict = torch.load(args["loadmodel"])
    model.load_state_dict(state_dict['state_dict'])

    image_dirs = [str(p) for p in Path(args["imagedir"]).glob("*") if p.is_dir()]
    total_psnr = 0
    for path in image_dirs:
        leftpath, rightpath, truthpath, outputimg = f'{path}/view1.png', f'{path}/view5.png', f'{path}/disp1.png', f'{path}/pred.png'
        truth = cv2.imread(truthpath, cv2.IMREAD_GRAYSCALE)

        # Run Stereo SGBM
        sgbm = run_stereo_sgbm(leftpath, rightpath)
        sgbm_inpaint_wls = postprocess_inpaint_wls(sgbm)

        # Run PSMNet
        disp = run_psmnet(model, leftpath, rightpath, outputimg)
        psmnet_fill = postprocess_fill_in_missing(source=sgbm_inpaint_wls, dest=disp.copy())
        psmnet_fill_wls = postprocess_wls_filter(psmnet_fill)
        
        # Replace output image with postprocessed output image
        cv2.imwrite(outputimg, psmnet_fill_wls) 

        psnr_score = calculate_psnr(truth, psmnet_fill_wls)
        total_psnr += psnr_score
        print(f"{path.split('/')[-1]}: {psnr_score}")

    print(f"Average PSNR Score: {total_psnr/len(image_dirs)}")