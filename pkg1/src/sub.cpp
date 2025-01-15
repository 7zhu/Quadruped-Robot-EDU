#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "chrono"
using namespace std::chrono_literals;
class TurtleCircleNode : public rclcpp::Node
{
public:
    explicit TurtleCircleNode(const std::string &node_name) : Node(node_name)
    {
        subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10, std::bind(&TurtleCircleNode::subTopic_callback_, this, std::placeholders::_1));
    }
    void subTopic_callback_(const geometry_msgs::msg::Twist::SharedPtr cmd)
    {
        RCLCPP_INFO(get_logger(), "now liner-x=%f", cmd->linear.x);
    }
private:
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscriber_;
};
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TurtleCircleNode>("turtle_circle");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
