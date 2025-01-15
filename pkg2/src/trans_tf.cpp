#include <rclcpp/rclcpp.hpp> // 包含ROS 2的C++客户端库
#include <geometry_msgs/msg/transform.hpp> // 包含变换消息类型
#include <tf2/LinearMath/Quaternion.h> // 包含四元数库，用于处理旋转
#include "tf2_geometry_msgs/tf2_geometry_msgs.h" // 包含tf2与geometry_msgs之间的转换函数
#include "tf2_ros/static_transform_broadcaster.h" // 包含静态变换广播器
#include "tf2_ros/transform_listener.h" // 包含变换监听器
#include "tf2_ros/buffer.h" // 包含tf2的缓存
#include "tf2/utils.h" // 包含tf2的工具函数
#include "chrono" // 包含C++标准库中的chrono时间库
using namespace std::chrono_literals; // 使用chrono库中的字面量

class DynamicTF : public rclcpp::Node // 定义DynamicTF类，继承自rclcpp::Node
{
public:
    DynamicTF(const std::string node_name) : Node(node_name) // 构造函数，初始化节点
    {
        this->broad_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this); // 创建一个静态变换广播器的共享指针
        timer_ = this->create_wall_timer(100ms, std::bind(&DynamicTF::PublishTF, this)); // 创建一个定时器，每100毫秒调用一次PublishTF函数
    }
    void PublishTF() // PublishTF成员函数，用于发布变换
    {
        geometry_msgs::msg::TransformStamped transform; // 创建一个变换消息
        transform.header.stamp = this->get_clock()->now(); // 设置变换的时间戳为当前时间
        transform.header.frame_id = "map"; // 设置父坐标系为“map”
        transform.child_frame_id = "target"; // 设置子坐标系为“target”
        transform.transform.translation.x = 5.0; // 设置在x轴上的平移距离为5.0
        transform.transform.translation.y = 5.0; // 设置在y轴上的平移距离为5.0
        transform.transform.translation.z = 5.0; // 设置在z轴上的平移距离为5.0
        tf2::Quaternion rotation; // 创建一个四元数对象
        rotation.setRPY(0.0, 0.0, 60 * M_PI / 180.0); // 设置四元数的旋转为绕z轴旋转60度
        transform.transform.rotation = tf2::toMsg(rotation); // 将四元数转换为消息类型，并设置到变换消息中
        this->broad_->sendTransform(transform); // 通过广播器发送变换消息
    };
private:
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> broad_; // 静态变换广播器的共享指针
rclcpp::TimerBase::SharedPtr timer_; // 定时器的共享指针
};
int main(int argc, char **argv) // main函数，程序的入口点
{
    rclcpp::init(argc, argv); // 初始化ROS 2
    auto node = std::make_shared<DynamicTF>("dynamicTF"); // 创建DynamicTF节点的共享指针
    rclcpp::spin(node); // 进入循环，等待回调函数被调用
    rclcpp::shutdown(); // 关闭ROS 2
return 0; // 程序结束
}
