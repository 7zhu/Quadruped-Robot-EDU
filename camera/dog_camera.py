import time
import sys
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Value
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import mediapipe as mp_mediapipe
import signal


# 初始化视频客户端
def initialize_video_client(network_interface):
    ChannelFactoryInitialize(0, network_interface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    return client


# 获取视频帧
def get_video_frame(client):
    code, data = client.GetImageSample()
    if code == 0:
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image
    else:
        print("Get image sample error. code:", code)
        return None



# 手势检测线程
def dog_camera(network_interface):
    # 在子进程中重新初始化
    client = initialize_video_client(network_interface)
    while True:
        # 获取当前时间
        current_time = time.time()
        image = get_video_frame(client)
        if image is None:
            break

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_image = cv2.bitwise_and(imgRGB, imgRGB)
        result = hands.process(masked_image)
        cv2.imshow("front_camera", image)
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyWindow("front_camera")


# 主函数
def main():    
    network_interface = sys.argv[1]  # 获取网络接口
    dog_camera(network_interface)
    


if __name__ == "__main__":
    main()
