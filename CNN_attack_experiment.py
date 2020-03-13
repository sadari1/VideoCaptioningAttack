import json
import os
import torch
import torch.nn as nn
import skvideo.io
import matplotlib.pyplot as plt
import math
import pretrainedmodels
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from pretrainedmodels import utils as ptm_utils
import torch.optim as optim
import PIL

BATCH_SIZE = 16
DIM = 224
c = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    convnet = "vgg16"

    use_carlini = True

    #0.2 seems to transfer to other models like resnet ("a red car") and vgg (seems to not mention cars but misclassifies). 0.005 and c = 10 transfers very well across other nasnet models.
    learning_rate = 0.2
    conv =  pretrainedmodels.__dict__[convnet](num_classes=1000, pretrained='imagenet')
    conv.eval()
    conv.to(device)

    video_name = "SaYwh6chmiw_15_40.avi"
    video_path = "D:\\College\\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\"
    frames = skvideo.io.vread(video_path+video_name,num_frames=BATCH_SIZE)[0:BATCH_SIZE]

    plt.imshow(frames[0])

    '''
    466 - 'bullet train, bullet',
    172: 'whippet',
    
    '''
    target_class = 172

    target = []
    for f in range(BATCH_SIZE):
        target.append(target_class)


     #this is a taxi
    target = np.array(target, dtype=np.float32)
    num_iterations = 1000
    if torch.cuda.is_available():
        delta = Variable(torch.zeros(frames.shape).cuda(), requires_grad=True)
    # else:
    #    delta = Variable(torch.zeros(frames.shape), requires_grad=True)

    optimizer = optim.Adam([delta],
                           lr=learning_rate,
                           betas=(0.9, 0.999))

    # target = np.zeros([1000], np.float32)
    # # for f in target:
    # np.put(target, 468, 1.00)

    # print(target, np.argmax(target), target.shape)
    print(target)

    tf_img_fn = ptm_utils.TransformImage(conv)
    load_img_fn = PIL.Image.fromarray


    original = original_create_batches(frames_to_do=frames, batch_size=BATCH_SIZE, tf_img_fn=tf_img_fn, load_img_fn=load_img_fn)
    # original = torch.Tensor(original)
    original = original[0].to(device)


    #705 is a passenger car (train)
    with torch.no_grad():
        original_output = conv(original)
        print(original_output, original_output.shape)
        for f in original_output:
            print(np.argmax(np.round(f)))

    frames = torch.Tensor(frames).cuda().float()
    loss = nn.CrossEntropyLoss()
    target = torch.Tensor(target).long().cuda()
    adversarial_frames = attack(delta, conv, num_iterations, frames, target, optimizer, loss, use_carlini).detach().cpu().numpy()

    plt.imshow(adversarial_frames[0])
    plt.show()

    np.save("{}{}CNN_{}.npy".format(video_path, convnet, video_name[:-4]), adversarial_frames)
    print("Saved at: {}{}CNN_{}.npy".format(video_path, convnet, video_name[:-4]))

    outputfile = video_path + "{}Adversarial_{}".format(convnet, video_name)
    writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
        # huffyuv is lossless. r10k is really good

        # '-c:v': 'libx264', #libx264 # use the h.264 codec
        '-c:v': 'huffyuv',  # r210 huffyuv r10k
        # '-pix_fmt': 'rgb32',
        # '-crf': '0', # set the constant rate factor to 0, which is lossless
        # '-preset': 'ultrafast'  # ultrafast, veryslow the slower the better compression, in princple, try
    })
    for f in adversarial_frames:
        writer.writeFrame(f)

    writer.close()

def detach(input):
    return input.detach().cpu().numpy()

def carliniwagner(output, target, k):

    #Top value and index of it
    values, indices = torch.topk(output, 1)

    k = torch.Tensor([k]).cuda()
    for f in range(0, len(output)):
        # if (indices[f].detach().cpu().numpy()[0] == target[f].detach().cpu().numpy()):


        #If the index of the max matches the target, then max the second highest logit
        if (detach(indices[f])[0] == detach(target[f])):
            # measured_value = (output[f][new_index[f].detach().cpu().numpy()[1]] - output[f][target[f].detach().cpu().numpy()])

            #Gets the top two values and their indices
            new_val, new_index = torch.topk(output, 2)

            #Now take second max logit - the logit for target

            # Technically this can just be output[f][detach(new_index[f])[1]] - values[f][0] since it's implied values[f][0] have the target indices.
            measured_value = (output[f][detach(new_index[f])[1]] - output[f][detach(target[f])])

            #Should clamp this so it's the max of the difference logits or -k
            # values[f] = torch.clamp(measured_value, min=-k)
            values[f] = torch.max(measured_value, -k)
            print(measured_value, values[f], -k)

        else:
            # measured_value = (output[f][indices[f].detach().cpu().numpy()[0]] - output[f][target[f].detach().cpu().numpy()])

            #Take max logit - logit for target
            measured_value = (output[f][detach(indices[f])[0]] - output[f][detach(target[f])])

            #Clamp this so it's the max of the difference in logits or -k
            # values[f] = torch.clamp(measured_value, min=-k)
            values[f] = torch.max(measured_value, -k)
            print(measured_value, values[f], -k)
    # print(values)

    #This isn't in the Carlini-Wagner attack but I did the mean of the CW result of each frame
    return values.mean(0)

        #max(output) making sure it's not target, - output[target]




def attack(delta, model, num_iterations, original, target, optimizer, loss, use_carlini):
    model.eval()
    dc = 255



    for i in range(num_iterations):

        print("Iteration {}".format(i))
        apply_delta = torch.clamp(delta * 255., min=-dc, max=dc)

        pass_in = torch.clamp(apply_delta + original, min=0, max=255)

        if(use_carlini):
            # pass_in = 0.5 * (torch_arctanh(pass_in / 255.) + 1)


            #Passing in 0.5( tanh(w) + 1) into the function
            pass_in = 0.5 * (torch_arctanh(pass_in / 255.) + 1)

            if i % 100 == 0:
                plt.imshow(pass_in[0].detach().cpu().numpy())
                plt.show()


            batch = create_batches(pass_in)
            output = model(batch)

            #This gets the actual output of the function
            cost = carliniwagner(output, target, 1)

            # First take 0.5(tanh(w) + 1) - x
            normterm = 0.5*((pass_in / 255.).tanh() + 1) - (original/255.)

            # Then take the l2 norm of the mean difference
            normterm = normterm.mean(0).norm()

            # Then square it element-wise
            print(normterm)
            # normterm = normterm.pow(2)

            cost = normterm + c * cost

            # plt.imshow(pass_in[0].detach().cpu().numpy())
            # plt.show()

            check = []
            for f in range(BATCH_SIZE):
                print(np.argmax(np.round(detach(output[f]))), np.round(detach(target[f])),
                      np.argmax(np.round(detach(output[f]))) == np.round(detach(target[f])))

                check.append(np.argmax(np.round(detach(output[f]))) == np.round(detach(target[f])))

            if (np.array(check).all() == True):
                print("Early stop at iteration {}".format(i))
                return pass_in

        else:


            batch = create_batches(pass_in)

            output = model(batch)



            cost = loss(output, target)
            # cost = loss(output, target)
            print("Loss: {}".format(cost))

            check = []
            for f in range(BATCH_SIZE):
                print(np.argmax(np.round(detach(output[f]))), np.round(detach(target[f])),
                  np.argmax(np.round(detach(output[f]))) == np.round(detach(target[f])))
                check.append(np.argmax(np.round(detach(output[f]))) == np.round(detach(target[f])))

            if( np.array(check).all() == True):
                print("Early stop at iteration {}".format(i))
                return pass_in

            # plt.imshow(pass_in[0].detach().cpu().numpy()/255.)
            # plt.show()
            # if (use_carlini):
            #minimize  magnitude of l2 norm of (1/2 tanh(w) + 1)  + c * f(1/2tanw(h)+1)
            #f is f(x') = max(max{Z(x')i : i!=t} - Z(x')t, -k)


            # essentially what f is supposed to be measuring is finding the maximum logits that predict another class other than the target t, subtract it from the logits for t, and max that value with -k.
            #f is basically the max of the max Z(x')i - Z(x')t and -k.
            #Z(x) = z is the logits, the output of all layers except for the softmax

                """
                objective function f is f(x') = max (  max( { Z(x')i : i!=t} - Z(x')t, -k)
                Z(x) = z is the output of all the layers except for the softmax layer, so the logits    
                
                
                so max( last layer logits for the input ) - last layer logits for the target class
                
                make sure argmax of output is not target
                max( max(output) - output[target], -k)
                
                the function to minimize is || 0.5 tanh(w) + 1 ||22 + c * f(0.5tanh(w) +1)
                so the magnitude of the l2norm of (0.5 tanh(w) + 1) + c * f( 0.5 tanh(w) + 1), where f is the function above
                """

            else:

                y = torch_arctanh(original / 255.).cuda()
                w = torch_arctanh(pass_in / 255.) - y
                normterm = ((w + y).tanh() - y.tanh())
                normterm = normterm.mean(0).norm()




                # normterm = self.delta / 255.
                # normterm = normterm.mean(0).norm()
                # cost = (c * cost.tanh() + 1) + ((1 - c) * normterm.mean(0).norm().tanh() + 1)
                print("Cost:\t{}\t+\tNormterm:\t{}".format(cost, normterm))
                cost = cost + (c * normterm)

            # calculate gradients
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
        print("\nIteration: {}, cost: {}".format(i, cost))
    print("Did not converge")



    return pass_in
#
# def loss(output, target):
#     # output_loss = []
#     # target = torch.Tensor(target).cuda().float()
#     # for f in output:
#     #     # _, pred = torch.max(f, 1)
#     loss = nn.CrossEntropyLoss(output, target)
#         # output_loss.append(loss)
#     return loss

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor



def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def create_batches(frames_to_do, batch_size=BATCH_SIZE):
    n = frames_to_do.shape[0]
    h, w = frames_to_do.shape[1:3]
    scale = 0.875
    input_size = [3, DIM, DIM]
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    input_range = [0, 1.]
    input_space = 'RGB'
    expand_size = int(math.floor(max(input_size) / scale))
    if w < h:
        ow = expand_size
        oh = int(expand_size * h / w)
    else:
        oh = expand_size
        ow = int(expand_size * w / h)

    tfs = []
    tfs.append(ToSpaceBGR(input_space == 'BGR'))
    tfs.append(ToRange255(max(input_range) == 255))
    tfs.append(transforms.Normalize(mean=mean, std=std))
    tf = transforms.Compose(tfs)

    a = int((0.5 * oh) - (0.5 * float(input_size[1])))
    b = a + input_size[1]
    c = int((0.5 * ow) - (0.5 * float(input_size[2])))
    d = c + input_size[2]

    if n < batch_size:
        # logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    # logger.info("Generating {} batches...".format(n // batch_size))

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx + batch_size, n)))

        # <batch, h, w, ch> <0,255>
        batch_tensor = frames_to_do[frames_idx]

        pass_in = batch_tensor.permute(0, 3, 1, 2) / 255.
        inp = torch.nn.functional.interpolate(pass_in,
                                              size=(oh, ow),
                                              mode='bilinear', align_corners=True)
        # Center cropping
        cropped_frames = inp[:, :, a:b, c:d]
        # cropped_image = cropped_image.contiguous()
        for i in range(len(cropped_frames)):
            cropped_frames[i] = tf(cropped_frames[i])
    return cropped_frames




def original_create_batches(frames_to_do, load_img_fn, tf_img_fn, batch_size=32):

    n = len(frames_to_do)
    if n < batch_size:
        print("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    print("Generating {} batches...".format(n // batch_size))
    batches = []
    frames_to_do = np.array(frames_to_do)

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx+batch_size, n)))
        batch_frames = frames_to_do[frames_idx]

        batch_tensor = torch.zeros((len(batch_frames),) + tuple(tf_img_fn.input_size))
        for i, frame_ in enumerate(batch_frames):
            input_img = load_img_fn(frame_)
            input_tensor = tf_img_fn(input_img)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            batch_tensor[i] = input_tensor

        batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(batch_ag)
    return batches




if __name__ == '__main__':

    main()