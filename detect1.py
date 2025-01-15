import os
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()#detect的绝对路径
ROOT = FILE.parents[0]  #副目录的路径，即yolov5的路径
if str(ROOT) not in sys.path:#确保root在模块查询路径中，以确保能够导包成功
    sys.path.append(str(ROOT))  #如果不在，则添加该路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径
from ultralytics.utils.plotting import Annotator, colors#导入必要的yolov5模块
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (check_img_size,check_imshow,check_requirements,cv2,non_max_suppression,scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode,time_sync
class YOLOdetect:
    def __init__(self,source,device="" ,weights=ROOT / "yolov5s.pt",data=ROOT / "data/YOLO_train.yaml",imgsz=(640, 640)):
        self.source=source#传入路径
        self.device=select_device(device)#选择默认device
        self.conf_thres=0.25#置信度
        self.iou_thres=0.45
        self.max_det=1000#最大框数
        self.line_thickness=3#框线粗细
        self.weights=weights
        self.data=data
        self.imgsz=imgsz
    def load(self):
        """加载模型与数据"""
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt#模型的步长，模型的类别名（就是检测目标的名字），模型是不是pytorch
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.dataset = LoadStreams(str(self.source), img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=1)#将视频流加载进数据集
    def drawing(self,det,im0,annotator):
        """画框和中心点
        Args:
             det:检测结果
             im0:原图
             annotator:标注器"""
        num=0
        for *xyxy, conf, cls in reversed(det):
            num+=1
            c = int(cls) 
            center_x=int((xyxy[0]+xyxy[2])/2)#中心点坐标
            center_y=int((xyxy[1]+xyxy[3])/2)
            cv2.circle(im0,(center_x,center_y),5,colors(c),7)
            cv2.putText(im0,f"({center_x},{center_y})",(center_x+5,center_y+5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
            label = f"{self.names[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))#给标注器赋值
            self.s+=f" No.{num} {self.names[int(c)]} " 
            self.s+=f"center:{(center_x,center_y)};"
    def runinference(self,pred,im0s,im,t1):
        """处理框架
        Args:
             pred:prediction,即预测结果
             im0s:原图
             im:预处理后的图像
             t1:每张图片开始处理时的时间戳"""
        for i, det in enumerate(pred):  #对得到的框进行处理
            self.seen+=1
            im0 = im0s[i].copy()
            self.s=f"{self.seen}: "
            self.s += "{:g}x{:g} ".format(*im.shape[2:])
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))#标注器
            if len(det):#判断有没有框,det储存的是每一张图的所有框
              det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()#恢复原图大小（坐标映射），方便能在原图画框
              for c in det[:, 5].unique():
                              n = (det[:, 5] == c).sum() 
                              self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, " #如果大于1就加s
            self.drawing(det,im0,annotator)
            im0 = annotator.result()#画上框
            time_pass=time_sync()-t1
            cv2.putText(im0,f"fps:{1/time_pass:.2f}",(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))#在左上角显示fps
            self.s+=f"time:{time_pass*1000:.2f}ms "
            self.s+=f"fps: {1/time_pass:.2f}"
            print(self.s)#在终端显示文本
            cv2.imshow("show",im0)
            cv2.waitKey(1)
    def inference(self):
         self.seen=0
         self.model.warmup(imgsz=(1 if self.pt  else self.bs, 3, *self.imgsz))# 给gpu热身         
         for path,im,im0s,vid_cap,s in self.dataset:#im0s是原图，s是图片的打印信息
             t1=time_sync()                                                                            
             im=torch.from_numpy(im).to(self.device)#im是resize之后的图片，把它从numpy数组转换为pytorch张量。
             im = im.half() if self.model.fp16 else im.float()
             im /= 255  # 0 - 255 to 0.0 - 1.0
             if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
             pred=self.model(im,augment=False,visualize=False)#推理出非常多框
             pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)#非极大值过滤，获得很多框的xyxy，置信度，检测出来的目标所属类别
             self.runinference(pred,im0s,im,t1)
             if cv2.waitKey(1) & 0xFF == ord('q'):#按q退出
                 cv2.destroyAllWindows()
                 return
    @smart_inference_mode()
    def run(self):
        self.load()
        t2=time_sync()
        self.inference()
        print(f"run time:{(time_sync()-t2):.2f}s")
        
def main():
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))#检测有没有依赖包
    yolo=YOLOdetect(source=0)
    yolo.run()
if __name__ == "__main__":
    main()


         
              
                     

                     

            


        
    




        


        
        
        
