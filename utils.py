
import cv2
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def convertFlowToImage(flow_x, flow_y, lowerBound, higherBound):
    def CAST(v, L, H):
        if(v > H): return 255
        elif(v < L): return 0
        else: return int(255 * (v - L) / (H - L))
    
    img_x = np.zeros((flow_x.shape[0], flow_y.shape[1]), dtype=np.uint8)
    img_y = np.zeros((flow_x.shape[0], flow_y.shape[1]), dtype=np.uint8)

    for i in range(0, flow_x.shape[0]):
        for j in range(0, flow_y.shape[1]):
            x = flow_x[i, j]
            y = flow_y[i, j]
            img_x[i, j] = CAST(x, lowerBound, higherBound)
            img_y[i, j] = CAST(y, lowerBound, higherBound)
    return img_x, img_y

def encodeFlowMap(flow_map_x, flow_map_y, bound, to_jpg):
    flow_img_x = np.zeros(flow_map_x.shape, dtype=np.int32)
    flow_img_y = np.zeros(flow_map_y.shape, dtype=np.int32)

    flow_img_x, flow_img_y = convertFlowToImage(flow_map_x, flow_map_y,-bound, bound)
    
    encoded_x = np.zeros(flow_img_x.shape, dtype=np.uint8)
    encoded_y = np.zeros(flow_img_y.shape, dtype=np.uint8)

    if(to_jpg == True):
        encoded_x = cv2.imencode(".jpg", flow_img_x)
        encoded_y = cv2.imencode(".jpg", flow_img_y)
    else:
        #encoded_x.resize(flow_img_x.total())
        #encoded_y.resize(flow_img_y.total())
        #memcpy(encoded_x.data(), flow_img_x.data, flow_img_x.total())
        #memcpy(encoded_y.data(), flow_img_y.data, flow_img_y.total())
        encoded_x = flow_img_x.copy()
        encoded_y = flow_img_y.copy()
    
    return encoded_x, encoded_y
    pass

def extract_optical_flow(fname, bound, extype, step, resize_height, resize_width):
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #print('[TestDebug] ', fname, frame_count)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
    #buffer = np.empty((frame_count, resize_height, resize_width, 3), np.dtype('float32'))
    output_x, output_y, output_img = list(), list(), list()

    capture_image, prev_image, capture_gray, prev_gray = None, None, None, None
    flow, flow_split = None, None

    count = 0
    initialized = False
    # read in each frame, one at a time into the numpy buffer array
    while (count < frame_count):
        retaining, frame = capture.read()
        # 如果frame是None的话，先跳过
        if(type(frame) == type(None)):
            continue
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        dense_optflow = cv2.DenseOpticalFlow()
        #opticalflow_farneback = cv2.FarnebackOpticalFlow()
        #optical_flow = cv2.DualTVL1OpticalFlow_create()


        # build mats for the first frame
        if(initialized == False):
            prev_image = frame.copy()
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            initialized = True
            pass
        elif(count % step == 0):
            capture_image = frame.copy()
            capture_gray = cv2.cvtColor(capture_image, cv2.COLOR_BGR2GRAY)

            if(extype == 0):
                #opticalflow_farneback.cacl(prev_gray, capture_gray, flow)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, capture_gray, flow, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN )
                pass
            elif(extype == 1):
                flow = dense_optflow.calc(prev_gray, capture_gray, flow)
                #flow = optical_flow.calc(prev_gray, capture_gray, flow)
                pass
            else:
                print("[Warning] Unknown optical method. Using Farneback")
                #opticalflow_farneback.cacl(prev_gray, capture_gray, flow)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, capture_gray, flow, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN )
                pass
            pass
        
            str_x, str_y, str_img = [], [], []
            flow_split = cv2.split(flow)
            str_x, str_y = encodeFlowMap(flow_split[0], flow_split[1], bound, True)
            str_img = cv2.imencode(".jpg", capture_image)

            output_x.append(str_x)
            output_y.append(str_y)
            output_img.append(str_img)

            # swap
            prev_gray, capture_gray = capture_gray, prev_gray
            prev_image, capture_image = capture_image, prev_image
            pass


        #if (frame_height != resize_height) or (frame_width != resize_width):
        #    frame = cv2.resize(frame, (resize_width, resize_height))
        #buffer[count] = frame
        count += 1

    # release the VideoCapture once it is no longer needed
    capture.release()

    return [output_x, output_y, output_img]

def saveImg(fname, img):
    f = open(fname, 'wb')
    for i in range(0, img.shape[0]):
        f.write(img[i])
    f.close()
