import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
import cv_bridge
import numpy as np
from eye_hand_calibration.srv import CaptureCalibrationData

class CalibrationNode(Node):
    """手眼标定核心节点，负责采集标定数据、触发标定计算"""
    def __init__(self):
        super().__init__('calibration_node')
        
        # 初始化参数
        self.bridge = cv_bridge.CvBridge()
        self.calibration_data_count = 0
        self.max_calibration_data = 12  # 采集12组数据（最优标定数量）
        
        # 创建服务客户端（调用标定数据采集服务）
        self.capture_client = self.create_client(
            CaptureCalibrationData,
            '/capture_calibration_data'
        )
        
        # 创建服务客户端（调用标定计算服务）
        self.calibrate_client = self.create_client(
            Trigger,
            '/calibrate'
        )
        
        # 创建订阅者（接收标定板图像）
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # 创建定时器（自动采集标定数据，每5秒一组）
        self.timer = self.create_timer(5.0, self.capture_calibration_data)
        
        self.get_logger().info("手眼标定节点已启动，开始采集标定数据...")

    def image_callback(self, msg):
        """接收相机图像，显示标定板视野"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow("Calibration Board View", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")

    def capture_calibration_data(self):
        """调用服务采集单组标定数据"""
        if self.calibration_data_count >= self.max_calibration_data:
            self.timer.cancel()  # 停止定时器
            self.start_calibrate()  # 触发标定计算
            return
        
        # 构造服务请求
        req = CaptureCalibrationData.Request()
        req.save_data = True
        
        # 等待服务可用
        if not self.capture_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("标定数据采集服务不可用")
            return
        
        # 发送服务请求
        future = self.capture_client.call_async(req)
        future.add_done_callback(self.capture_callback)

    def capture_callback(self, future):
        """标定数据采集回调"""
        try:
            resp = future.result()
            if resp.success:
                self.calibration_data_count += 1
                self.get_logger().info(f"已采集 {self.calibration_data_count}/{self.max_calibration_data} 组标定数据")
            else:
                self.get_logger().error(f"采集失败: {resp.message}")
        except Exception as e:
            self.get_logger().error(f"服务调用失败: {str(e)}")

    def start_calibrate(self):
        """触发手眼标定计算"""
        self.get_logger().info("开始执行手眼标定计算...")
        
        # 构造服务请求
        req = Trigger.Request()
        
        # 等待服务可用
        if not self.calibrate_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("标定计算服务不可用")
            return
        
        # 发送服务请求
        future = self.calibrate_client.call_async(req)
        future.add_done_callback(self.calibrate_callback)

    def calibrate_callback(self, future):
        """标定计算回调"""
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info("手眼标定完成，结果已保存至TF变换")
            else:
                self.get_logger().error(f"标定失败: {resp.message}")
            cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f"标定服务调用失败: {str(e)}")

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = CalibrationNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()