import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from pclpy import pcl

class PointCloudFilterNode(Node):
    """点云滤波节点，采用直通滤波+统计滤波去除噪声"""
    def __init__(self):
        super().__init__('point_cloud_filter_node')
        
        # 创建订阅者（接收原始点云）
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.point_cloud_callback,
            10
        )
        
        # 创建发布者（发布滤波后点云）
        self.filtered_pub = self.create_publisher(
            PointCloud2,
            '/filtered_points',
            10
        )
        
        # 滤波参数配置
        self.z_min = 0.75  # 桌面最低高度
        self.z_max = 0.85  # 桌面最高高度
        self.mean_k = 20   # 统计滤波邻域点数量
        self.std_dev_thresh = 1.0  # 统计滤波标准差阈值
        
        self.get_logger().info("点云滤波节点已启动，开始处理点云数据...")

    def point_cloud_callback(self, msg):
        """点云数据回调处理函数"""
        try:
            # 1. 解析ROS2 PointCloud2数据为numpy数组
            points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not points_list:
                self.get_logger().warn("接收到空点云数据")
                return
            points_np = np.array(points_list, dtype=np.float32)
            
            # 2. 转换为PCL点云格式
            pcl_cloud = pcl.PointCloud.PointXYZ()
            pcl_cloud.from_array(points_np)
            
            # 3. 直通滤波：保留Z轴范围内的点（过滤桌面上下无关区域）
            pass_through = pcl.filters.PassThrough.PointXYZ()
            pass_through.setInputCloud(pcl_cloud)
            pass_through.setFilterFieldName("z")
            pass_through.setFilterLimits(self.z_min, self.z_max)
            cloud_pass = pcl.PointCloud.PointXYZ()
            pass_through.filter(cloud_pass)
            
            # 4. 统计滤波：去除孤立噪声点
            stat_filter = pcl.filters.StatisticalOutlierRemoval.PointXYZ()
            stat_filter.setInputCloud(cloud_pass)
            stat_filter.setMeanK(self.mean_k)
            stat_filter.setStddevMulThresh(self.std_dev_thresh)
            cloud_filtered = pcl.PointCloud.PointXYZ()
            stat_filter.filter(cloud_filtered)
            
            # 5. 转换回ROS2 PointCloud2格式并发布
            filtered_points = cloud_filtered.to_array()
            filtered_msg = pc2.create_cloud_xyz32(msg.header, filtered_points)
            self.filtered_pub.publish(filtered_msg)
            
        except Exception as e:
            self.get_logger().error(f"点云滤波处理失败: {str(e)}")

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = PointCloudFilterNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()