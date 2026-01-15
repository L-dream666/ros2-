import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PointStamped, Point
from tf2_ros import TransformListener, Buffer
from tf2_ros.exceptions import TransformException
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import cv_bridge
import sensor_msgs_py.point_cloud2 as pc2

class ObjectLocalizationNode(Node):
    """物体定位节点，实现颜色识别、形状匹配与坐标转换"""
    def __init__(self):
        super().__init__('object_localization_node')
        
        # 初始化工具与参数
        self.bridge = cv_bridge.CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 颜色阈值配置（HSV空间）
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 50, 50])
        self.red_upper2 = np.array([179, 255, 255])
        self.yellow_lower = np.array([25, 50, 50])
        self.yellow_upper = np.array([35, 255, 255])
        
        # 形状筛选参数
        self.apple_circularity_min = 0.85  # 苹果圆度阈值
        self.banana_aspect_ratio_min = 2.8  # 香蕉长宽比阈值
        
        # 创建订阅者
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/filtered_points',
            self.point_cloud_callback,
            10
        )
        
        # 创建发布者（发布物体三维坐标）
        self.object_pub = self.create_publisher(
            PointStamped,
            '/object_position',
            10
        )
        
        # 缓存点云数据
        self.latest_point_cloud = None
        
        self.get_logger().info("物体定位节点已启动，开始识别与定位目标物体...")

    def image_callback(self, msg):
        """RGB图像回调，实现颜色识别与形状匹配"""
        try:
            # 1. 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 2. 颜色分割（提取红色苹果与黄色香蕉）
            mask_red1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
            mask_red2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_yellow = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
            
            # 3. 形态学操作去除噪声
            kernel = np.ones((5, 5), np.uint8)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            
            # 4. 轮廓检测与形状筛选
            self.detect_and_localize_object(mask_red, "apple", msg.header)
            self.detect_and_localize_object(mask_yellow, "banana", msg.header)
            
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")

    def detect_and_localize_object(self, mask, obj_type, header):
        """检测目标物体并完成定位"""
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 计算轮廓面积，过滤小轮廓
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            # 形状筛选
            if not self.check_shape(contour, obj_type):
                continue
            
            # 计算轮廓中心（图像坐标系）
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 从点云中获取三维坐标并转换
            self.get_3d_position(cX, cY, header)

    def check_shape(self, contour, obj_type):
        """形状匹配验证"""
        # 计算轮廓近似
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if obj_type == "apple":
            # 计算圆度
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (peri ** 2)
            return circularity >= self.apple_circularity_min
        
        elif obj_type == "banana":
            # 计算外接矩形与长宽比
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            return aspect_ratio >= self.banana_aspect_ratio_min or 1/aspect_ratio >= self.banana_aspect_ratio_min
        
        return False

    def point_cloud_callback(self, msg):
        """缓存最新点云数据"""
        self.latest_point_cloud = msg

    def get_3d_position(self, cX, cY, header):
        """从点云中提取三维坐标并完成坐标系转换"""
        if self.latest_point_cloud is None:
            self.get_logger().warn("无可用点云数据，无法获取三维坐标")
            return
        
        try:
            # 1. 从点云中提取对应像素的三维坐标（相机坐标系）
            points = list(pc2.read_points(
                self.latest_point_cloud,
                field_names=("x", "y", "z"),
                uvs=[(cX, cY)],
                skip_nans=True
            ))
            
            if not points:
                return
            object_cam = np.array(points[0], dtype=np.float32)
            
            # 2. 查询TF变换（相机坐标系→机器人基坐标系）
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_link',
                rclpy.time.Time()
            )
            
            # 3. 解析平移与旋转矩阵
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            
            q = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            rot = R.from_quat(q)
            rot_matrix = rot.as_matrix()
            
            # 4. 坐标转换：相机坐标系→机器人基坐标系
            object_base = rot_matrix @ object_cam + np.array([tx, ty, tz])
            
            # 5. 发布物体三维坐标
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.header.frame_id = 'base_link'
            point_stamped.point.x = object_base[0]
            point_stamped.point.y = object_base[1]
            point_stamped.point.z = object_base[2]
            
            self.object_pub.publish(point_stamped)
            self.get_logger().info(f"已定位物体，坐标：({object_base[0]:.3f}, {object_base[1]:.3f}, {object_base[2]:.3f})")
            
        except TransformException as e:
            self.get_logger().error(f"TF变换失败: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"三维坐标获取失败: {str(e)}")

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = ObjectLocalizationNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()