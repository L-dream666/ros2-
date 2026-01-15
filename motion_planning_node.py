import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from scipy.interpolate import interp1d

class MotionPlanningNode(Node):
    """运动规划节点，生成双臂协作平滑运动轨迹"""
    def __init__(self):
        super().__init__('motion_planning_node')
        
        # 机器人参数配置
        self.left_arm_joints = [
            'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
            'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6'
        ]
        self.right_arm_joints = [
            'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3',
            'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6'
        ]
        self.max_joint_vel = 0.4  # 最大关节速度 (rad/s)
        self.max_joint_acc = 0.25  # 最大关节加速度 (rad/s²)
        self.trajectory_time = 5.0  # 轨迹总时长 (s)
        
        # 创建订阅者（接收物体坐标）
        self.object_sub = self.create_subscription(
            PointStamped,
            '/object_position',
            self.object_callback,
            10
        )
        
        # 创建发布者（发布关节轨迹指令）
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )
        
        self.get_logger().info("运动规划节点已启动，等待物体坐标信息...")

    def object_callback(self, msg):
        """接收物体坐标，生成双臂运动轨迹"""
        try:
            # 1. 提取物体坐标（机器人基坐标系）
            obj_x = msg.point.x
            obj_y = msg.point.y
            obj_z = msg.point.z
            
            self.get_logger().info(f"接收到物体坐标：({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})")
            
            # 2. 生成左臂（主抓取）运动轨迹
            left_trajectory = self.generate_arm_trajectory(obj_x, obj_y, obj_z, "left")
            
            # 3. 生成右臂（辅助支撑）运动轨迹
            right_trajectory = self.generate_arm_trajectory(obj_x + 0.05, obj_y - 0.05, obj_z, "right")
            
            # 4. 合并并发布轨迹
            combined_trajectory = self.combine_trajectories(left_trajectory, right_trajectory)
            self.trajectory_pub.publish(combined_trajectory)
            
        except Exception as e:
            self.get_logger().error(f"运动轨迹生成失败: {str(e)}")

    def generate_arm_trajectory(self, target_x, target_y, target_z, arm_type):
        """生成单臂五次多项式插值轨迹"""
        # 模拟关节初始位置（实际应从机器人获取当前关节状态）
        initial_joints = np.array([0.0, -np.pi/4, 0.0, np.pi/4, 0.0, 0.0])
        
        # 模拟目标关节位置（实际应通过运动学逆解计算）
        # 此处简化：根据物体坐标映射关节位置（实际项目需使用KDL或MoveIt2进行逆解）
        target_joints = self.inverse_kinematics(target_x, target_y, target_z, arm_type)
        
        # 生成时间点
        time_points = np.linspace(0, self.trajectory_time, num=50)
        
        # 五次多项式插值生成平滑轨迹
        trajectory_points = []
        for t in time_points:
            # 五次多项式插值（简化实现，保证位置、速度、加速度连续）
            s = self.fifth_order_polynomial(t, self.trajectory_time)
            joint_pos = initial_joints + s * (target_joints - initial_joints)
            joint_vel = np.zeros_like(joint_pos)  # 简化：实际应计算插值速度
            joint_acc = np.zeros_like(joint_pos)  # 简化：实际应计算插值加速度
            
            trajectory_point = JointTrajectoryPoint()
            trajectory_point.time_from_start = rclpy.duration.Duration(seconds=t).to_msg()
            trajectory_point.positions = joint_pos.tolist()
            trajectory_point.velocities = joint_vel.tolist()
            trajectory_point.accelerations = joint_acc.tolist()
            
            trajectory_points.append(trajectory_point)
        
        return trajectory_points

    def fifth_order_polynomial(self, t, T):
        """五次多项式插值函数（s(t)）"""
        if t >= T:
            return 1.0
        return 10 * (t/T)**3 - 15 * (t/T)**4 + 6 * (t/T)**5

    def inverse_kinematics(self, x, y, z, arm_type):
        """简化逆运动学求解（实际项目需替换为真实逆解）"""
        # 此处为模拟结果，仅用于仿真演示
        joint_angles = np.array([
            np.arctan2(y, x),
            -np.pi/3,
            np.pi/6,
            np.pi/4,
            np.pi/6,
            0.0
        ])
        
        if arm_type == "right":
            joint_angles[0] *= -1  # 右臂对称调整
        
        return joint_angles

    def combine_trajectories(self, left_trajectory, right_trajectory):
        """合并左右臂轨迹，发布至机器人"""
        combined_trajectory = JointTrajectory()
        combined_trajectory.header.frame_id = 'base_link'
        combined_trajectory.joint_names = self.left_arm_joints + self.right_arm_joints
        
        # 确保左右臂轨迹时间点一致
        min_length = min(len(left_trajectory), len(right_trajectory))
        for i in range(min_length):
            combined_point = JointTrajectoryPoint()
            combined_point.time_from_start = left_trajectory[i].time_from_start
            
            # 合并关节位置、速度、加速度
            combined_point.positions = left_trajectory[i].positions + right_trajectory[i].positions
            combined_point.velocities = left_trajectory[i].velocities + right_trajectory[i].velocities
            combined_point.accelerations = left_trajectory[i].accelerations + right_trajectory[i].accelerations
            
            combined_trajectory.points.append(combined_point)
        
        return combined_trajectory

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = MotionPlanningNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()