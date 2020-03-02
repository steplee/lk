import numpy as np
import cv2
import torch, torch.nn.functional as F, torch.nn as nn
import sys, os, time

'''
Lucas-Kanade using pure pytorch.

Thinking before I implement let alone benchmark, I'm pretty sure
that grid_sample() will be the bottleneck. I haven't looked at how
THCTensor is structured, but it probably can't be stored in GPU texture
memory, which we can use hardware to bilinearly sample from for maximum
efficiency, especially considering we warp many times with the same image.
'''

laplacianSpatialFilter = torch.from_numpy(np.array((
    (1,1,1),
    (1,-8,1),
    (1,1,1)), dtype=np.float32)).unsqueeze_(0).unsqueeze_(0).cuda()
gaussian5_filter = torch.arange(5).repeat(5).view(5,5)
gaussian5_filter = torch.stack((gaussian5_filter,gaussian5_filter.t()), -1)
gaussian5_filter = (1./(2.*np.pi)) * torch.exp(-torch.sum((gaussian5_filter-2)**2., -1) / (2.))
gaussian5_filter /= gaussian5_filter.sum()
gaussian5_filter = gaussian5_filter.view(1,1,5,5).cuda()
#import cv2; cv2.imshow('gauss5',gaussian5_filter[0,0].cpu().numpy()); cv2.waitKey(0)

# TODO: Experiment to see if unsqueeze_ or view_ is faster or if they do the same under the hood.
#       Also if slicing into first channel allocates, it would be worth it to refactor to BCHW format.
def laplacian_spatial_domain(x):
    return F.conv2d(x.unsqueeze_(1), laplacianSpatialFilter, padding=1)[:,0] # add/remove channel dim

def gaussian5_spatial_domain(x):
    return F.conv2d(x.unsqueeze_(1), gaussian5_filter, padding=2)[:,0] # add/remove channel dim

def color_normalize_img(a):
    r = np.copy(a); r[r>0] = 0; r = abs(r)
    g = np.copy(a); g[g<0] = 0
    b = np.copy(a); b *= 0
    aa = np.stack( (b,g,r), -1)
    max_ = aa.max()
    return (255*(aa)/(max_)).astype(np.uint8)
def show_imgs(a, name='a'):
    if isinstance(a, torch.Tensor): a = a.cpu().detach().numpy()
    if len(a.shape) == 4: assert(a.shape[1] == 1); a = a[:,0]
    if len(a.shape) == 3: a = np.vstack(tuple(color_normalize_img(b) for b in a))
    else: a = color_normalize_img(a)
    cv2.imshow(name, a)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()


def Sobel(k, device=None):
    assert(k==3)
    s = nn.Conv2d(1,2,3,padding=1, bias=False)
    s.requires_grad_(False)
    s.weight.data = torch.FloatTensor([
        [[[1,0,-1], [2,0,-2], [1,0,-1]]],
        [[[1,2,1], [0,0,0],  [-1,-2,-1]]],
        ]).to(device=device)
    return s
_sobel3 = Sobel(3).cuda()

def homogeneousize(a):
    return np.concatenate( (a, np.ones((*a.shape[:-1],1),dtype=a.dtype)) , -1 )
def homogeneousizeMatrix(a):
    r,c = a.shape
    out = torch.eye(max(r,c), device=a.device)
    out[:r,:c] = a
    return out

# Assumes top 2x2 block is orthogonal, which is actually not the case
# considering we add an increment. Hopefully accurate enough though.
def fastInverse3x3(a):
    # Very fast, very accurate. Ignore below code.
    return torch.inverse(a.cpu()).to(a.device)
    # Very S L O W, but most accurate
    #return torch.inverse(a)

    a = a.cpu()
    o11,o12 = 1./a[0,0], -a[0,1]
    o21,o22 = -a[1,0], 1./a[1,1]
    return torch.cuda.FloatTensor([
        [o11,o12, -(o11-o21)*a[0,2]],
        [o21,o22,  (o12-o22)*a[1,2]],
        [0,0,1]]).view(3,3)


    #return torch.inverse(a)
    a = a.cpu()
    out = torch.zeros(a.shape, device=a.device)
    #out[:2,:2] = a[:2,:2].T
    out[0,0] = 1./a[0,0]
    out[1,1] = 1./a[1,1]
    out[0,1] = -a[0,1]
    out[1,0] = -a[1,0]
    out[:2,2] = -out[:2,:2].T @ a[:2,2]
    out[2,2] = 1
    #print('a\n',a,'\ninv\n',torch.inverse(a),'\nmy inv\n', out)
    return out.cuda()

def jac_aff_batched(x):
    J = torch.empty((2,6,x.shape[0]), dtype=torch.float32, device=x.device)
    J[0,0] = x[:, 0]
    J[0,1] = 0
    J[0,2] = x[:, 1]
    J[0,3] = 0
    J[0,4] = 1
    J[0,5] = 0

    J[1,0] = 0
    J[1,1] = x[:, 0]
    J[1,2] = 0
    J[1,3] = x[:, 1]
    J[1,4] = 0
    J[1,5] = 1
    return J

# grid_for_image() and warp_aff_space() are split because the grid can be re-used
# for all iterations.
def grid_for_image(x):
    print(x.shape)
    h,w = x.size(-2), x.size(-1)
    u,v = 1, h/w
    i = np.stack( np.meshgrid(np.linspace(-u,u,w), np.linspace(-v,v,h)) , -1 )
    i = homogeneousize(i)
    i = torch.from_numpy(i).to(torch.float32).to(x.device)
    return i
def warp_aff_space(grid,W):
    h,w = grid.size(0), grid.size(1)
    W = W.clone()
    W[1].mul_(w/h)
    i = grid @ W.T
    #i[...,1].mul_(w/h)
    #i = i.unsqueeze_(0).repeat(grid.size(0),1,1,1)
    i.unsqueeze_(0)
    return i
def warp_aff(x, W, grid):
    i = warp_aff_space(grid,W)
    #i = i.repeat(x.size(0),1,1,1)
    return F.grid_sample(x, i)

'''
Implements both the additive + compositional forward algorithms.
The composition formulation does less work per iteration: the Jacobian is evaluated
at the identity warp every time, which is indendent of 'p' and can be done once at the outset.
They are mathematically equivalent, but I haven't tested which works better in practice.
'''
def lk_affine(qrimg):
    # Scale is important, isn't the Hessian supposed to make it not so?
    # I think it has to do with a number <1 being squared is smaller, but >1 is larger.
    #DIV_BY = 1.
    DIV_BY = 255
    # For regularization, add diag(d) to Hessian.
    #DAMPENING = 128**2
    DAMPENING = .2
    DAMPENING = 4
    DIV_STD = True

    RECOMPUTE_GRAD = False

    COMPOSITIONAL = True
    ADDITIVE = not COMPOSITIONAL

    with torch.no_grad():
        h,w = qrimg.shape[-2:]
        assert(w >= h)
        aspect_hw = h/w
        u,v = 1, 1
        B = 2
        qrimg = qrimg / DIV_BY
        qrimg.sub_(qrimg.mean()) # Only subtract indepedent of batch dim (brightness constancy)
        if DIV_STD: qrimg.div_(qrimg.std())
        #qrimg = F.instance_norm(qrimg) # Don't do for LK (brightness constancy) but do for ECC

        grads = _sobel3(qrimg)
        W = torch.cuda.FloatTensor([[1,0,0], [0,1,0]])

        if COMPOSITIONAL:
            pix_fp_i = warp_aff_space(qrimg[0:1],W)[0].view(-1,2)
            J = jac_aff_batched(pix_fp_i)

        st = time.time()
        for iters in range(30):
            wqimg = warp_aff(qrimg[0:1], W)
            wgrads = _sobel3(wqimg[0:1]) if RECOMPUTE_GRAD else warp_aff(grads[:1], W)
            err = (wqimg[0] - qrimg[1]).squeeze()

            if ADDITIVE:
                pix_fp_i = warp_aff_space(qrimg[0:1],W)[0].view(-1,2)
                J = jac_aff_batched(pix_fp_i)
            print('J',J.shape)

            #wgrads = wgrads[0].permute(1,2,0).view(-1, 2)
            wgrads = wgrads[0].view(2, 1,-1)
            print('wg',wgrads.shape)
            sd = (J[0]*wgrads[0] + J[1]*wgrads[1]).T
            print(sd.shape)
            hess = (sd.unsqueeze(-1) @ sd.unsqueeze(1))
            hess = hess.mean(0)
            hess += torch.eye(6).to(hess.device) * DAMPENING
            print('Hessian', hess)

            print('err', err.shape)
            Jres = (sd * err.view(-1,1)).mean(0)
            print('Jres', Jres)

            inc,_ = torch.solve(Jres.unsqueeze_(1), hess.unsqueeze_(0))
            inc = inc.reshape(3,2).T
            print('Increment:\n',inc)

            if COMPOSITIONAL:
                inc[0,0] += 1
                inc[1,1] += 1
                W = (homogeneousizeMatrix(W) @ homogeneousizeMatrix(inc))[:2]
            else:
                W = W + inc
            w = warp_aff(qrimg[0:1], W)

            err2 = w[0] - qrimg[1]
            print('PreError ', err.view(-1).norm().item())
            print('PostError', err2.view(-1).norm().item())

            ##show_imgs(sd[:,0:3].view(180,322,3).permute(2,0,1),'sd')
            w = torch.cat((w,qrimg[1:2],err2.unsqueeze_(0).abs()))
            show_imgs(w,'warped')
        print(' - Forward Additive LK: {} iters in {}s.'.format(iters,time.time()-st))

'''
Inverse Compositional algorithm.
It is faster-per-loop for two reasons:
    1) It avoids recomputing the Jacobian. The 'composition' part takes care of this,
       the Jacobian is invariant to 'p' and only depends on 'x' (original pixel location).
       Note: The forward compositional approach also achieves this.
    2) It avoids recomputing the Hessian, by expressing 'steepest descent images' in terms
       of the template, which is invariant to 'p'.
'''
def lk_ic_affine(qrimg):
    DAMPENING = 4
    with torch.no_grad():
        h,w = qrimg.shape[-2:]
        qrimg = qrimg/255.
        assert(w >= h)
        aspect_hw = h/w
        qrimg.sub_(qrimg.mean()) # Only subtract indepedent of batch dim (brightness constancy)
        qrimg.div_(qrimg.std())

        grads = _sobel3(qrimg)

        W = torch.cuda.FloatTensor([[1,0,0], [0,1,0]])

        grid_fp = grid_for_image(qrimg).cuda()
        print('grid_fp',grid_fp.shape)

        pix_fp_i = warp_aff_space(grid_fp,W)[0].view(-1,2)
        J = jac_aff_batched(pix_fp_i)
        print('J',J.shape)

        rgrad = grads[1].view(2, 1,-1)
        print('rgrad',rgrad.shape)
        sd = (J[0]*rgrad[0] + J[1]*rgrad[1]).T
        print('sd', sd.shape)

        hess = (sd.unsqueeze(-1) @ sd.unsqueeze(1))
        hess = hess.mean(0)
        hess += torch.eye(6, device=hess.device) * DAMPENING
        print('hess', hess.shape)


        '''
        I had to negate wrt the paper. Either I could negate 'err' (= negate Jres, = negate sd),
        OR I had to not use inverse of the warp. I think not using inverse is incorrect (though it
        still works), so I stick with negating err (even though this is not specified in 'LK 20 years on')
        '''
        st = time.time()
        for iters in range(20):
            wqimg = warp_aff(qrimg[0:1], W, grid_fp)
            #err = -(wqimg[0] - qrimg[1]).squeeze()
            err = (qrimg[1] - wqimg[0])

            Jres = (sd * err.view(-1,1)).mean(0)

            # If you don't include this, profiler will report more total time. I guess it
            # counts double time when we call cpu() and the cuda stream hasn't finished.
            # Note: it doesn't *actually* increase wall speed.
            #torch.cuda.synchronize()

            inc,_ = torch.solve(Jres.cpu().unsqueeze_(1), hess.cpu().unsqueeze_(0))
            inc = inc.reshape(3,2).T
            inc[0,0].add_(1)
            inc[1,1].add_(1)
            #W = (homogeneousizeMatrix(W) @ torch.inverse(homogeneousizeMatrix(inc)))[:2]
            W = (homogeneousizeMatrix(W.cpu()) @ fastInverse3x3(homogeneousizeMatrix(inc)))[:2].cuda()

            w = warp_aff(qrimg[0:1], W, grid_fp)
            err2 = w[0] - qrimg[1]
            #print('PreError ', err.view(-1).norm().cpu().item())
            print('PostError', err2.view(-1).norm().cpu().item())

            ##show_imgs(sd[:,0:3].view(180,322,3).permute(2,0,1),'sd')
            w = torch.cat((w,qrimg[1:2],err2.unsqueeze_(0).abs()))
            show_imgs(w,'warped')
        print(' - Inverse Compositional LK: {} iters in {}s.'.format(iters,time.time()-st))

img1 = cv2.imread('/home/slee/Pictures/khopframe1.png', 0)
img1 = cv2.pyrDown(img1)
img1 = cv2.pyrDown(img1)
#img1 = cv2.pyrDown(img1); img1 = cv2.pyrUp(img1)
img1 = cv2.GaussianBlur(img1,(5,5),2)
m = np.eye(3)[:2]; m[:2,2] = (15,15)
img2 = cv2.warpAffine(img1, m, img1.shape[:2][::-1])
img1,img2 = img2,img1

qrimg = torch.from_numpy(np.stack((img1,img2))).cuda().unsqueeze_(1).to(torch.float32)
qrimg[1].add_(torch.randn_like(qrimg[1])*5).mul_(.99)

#lk_affine(img1,img2)
lk_ic_affine(qrimg)
lk_ic_affine(qrimg)
with torch.autograd.profiler.profile() as prof:
    for i in range(10):
        lk_ic_affine(qrimg)

print(prof.key_averages().table())
