import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2
# display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
img =jetson.utils.loadImage("/home/nvidia/Desktop/dog.34.jpg")

detections = net.Detect(img)
print(detections[0])
