import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import time

class GripperControlNode(Node):
    """夹爪控制节点，实现位置-力混合控制"""
    def __init__(self):
        super().__init__('gripper_control_node')
        
        # 夹爪参数配置
        self.gripper_open_angle = 1.2  # 夹爪完全张开角度 (rad)
        self.gripper_close_angle = 0.3  # 夹爪闭合目标角度 (rad)
        self.max_gripper_force = 0.5    # 最大夹持力 (N)
        self.gripper_joint_name = 'left_gripper_joint'
        
        # 抓取状态机
        self.grab_state = "IDLE"  # IDLE → OPEN → READY → CLOSE → GRAB_SUCCESS → RELEASE
        self.object_position = None
        
        # 创建订阅者
        self.object_sub = self.create_subscription(
            PointStamped,
            '/object_position',
            self.object_callback,
            10
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # 创建发布者（发布夹爪控制指令）
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_command',
            10
        )
        
        # 创建定时器（状态机更新，100ms一次）
        self.state_timer = self.create_timer(0.1, self.update_grab_state)
        
        self.get_logger().info("夹爪控制节点已启动，等待抓取指令...")

    def object_callback(self, msg):
        """接收物体坐标，触发抓取流程"""
        self.object_position = msg.point
        if self.grab_state == "IDLE":
            self.grab_state = "OPEN"
            self.get_logger().info("检测到目标物体，启动抓取流程")

    def joint_state_callback(self, msg):
        """接收关节状态，监测夹爪力与位置"""
        try:
            # 查找夹爪关节索引
            gripper_idx = msg.name.index(self.gripper_joint_name)
            current_angle = msg.position[gripper_idx]
            
            # 简化力检测：实际应通过力传感器数据，此处通过关节位置偏差估算
            current_force = abs(current_angle - self.gripper_close_angle) * 10.0
            if current_force > self.max_gripper_force:
                self.get_logger().warn(f"夹持力超限: {current_force:.2f}N > {self.max_gripper_force}N")
                
        except ValueError:
            pass
        except Exception as e:
            self.get_logger().error(f"关节状态解析失败: {str(e)}")

    def update_grab_state(self):
        """更新抓取状态机，执行对应动作"""
        if self.grab_state == "IDLE":
            return
        
        elif self.grab_state == "OPEN":
            # 发布夹爪张开指令
            self.publish_gripper_command(self.gripper_open_angle)
            time.sleep(0.5)  # 等待动作完成（简化）
            self.grab_state = "READY"
            self.get_logger().info("夹爪已张开，等待机械臂到位")
        
        elif self.grab_state == "READY":
            # 等待机械臂到达物体上方（简化：直接进入闭合流程）
            # 实际应通过机械臂轨迹完成信号触发
            self.grab_state = "CLOSE"
            self.get_logger().info("机械臂已到位，开始闭合夹爪")
        
        elif self.grab_state == "CLOSE":
            # 发布夹爪闭合指令（位置控制）
            self.publish_gripper_command(self.gripper_close_angle)
            time.sleep(1.0)  # 等待动作完成（简化）
            self.grab_state = "GRAB_SUCCESS"
            self.get_logger().info("夹爪闭合完成，抓取成功")
        
        elif self.grab_state == "GRAB_SUCCESS":
            # 抓取成功，等待释放指令（简化：延时后自动释放）
            time.sleep(5.0)
            self.grab_state = "RELEASE"
            self.get_logger().info("到达目标位置，准备释放物体")
        
        elif self.grab_state == "RELEASE":
            # 发布夹爪张开指令，释放物体
            self.publish_gripper_command(self.gripper_open_angle)
            time.sleep(0.5)
            self.grab_state = "IDLE"
            self.object_position = None
            self.get_logger().info("物体已释放，抓取流程结束，返回空闲状态")

    def publish_gripper_command(self, target_angle):
        """发布夹爪控制指令"""
        gripper_cmd = Float64MultiArray()
        gripper_cmd.data = [target_angle]
        self.gripper_pub.publish(gripper_cmd)

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = GripperControlNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()