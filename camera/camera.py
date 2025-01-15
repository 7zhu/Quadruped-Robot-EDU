import pyrealsense2 as rs
import numpy as np
import cv2
#==========================================================================================#
# detect()
# 检测的主题函数
#==========================================================================================#
def detect(save_img=False):
    # start depth camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,30)
    pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)  #对齐rgb和深度图

    while True:
        
        frames = pipeline.wait_for_frames() #等待最新的影像，wait_for_frames返回的是一個合成的影像
        frames = align_to_color.process(frames) #将上图获取视频帧对齐

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    # while out



#==========================================================================================#
# man!!!
#==========================================================================================#
def main():
    detect()

if __name__ == '__main__':
    main()
    