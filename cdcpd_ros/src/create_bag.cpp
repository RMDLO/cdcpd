#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <map>
#include <sstream>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/simple_filter.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <arc_utilities/ros_helpers.hpp>
#include <cdcpd/cdcpd.h>
#include <cdcpd/optimizer.h>

#include <opencv2/core/eigen.hpp>

#include <cdcpd_ros/Float32MultiArrayStamped.h>
#include <victor_hardware_interface/Robotiq3FingerStatus_sync.h>
#include <victor_hardware_interface/Robotiq3FingerStatus.h>

victor_hardware_interface::Robotiq3FingerStatus_sync::Ptr l_msg;
victor_hardware_interface::Robotiq3FingerStatus_sync::Ptr r_msg;

static victor_hardware_interface::Robotiq3FingerStatus_sync::Ptr gripper_status_origin_to_sync(
	const victor_hardware_interface::Robotiq3FingerStatus::ConstPtr origin, int diff)
{
	victor_hardware_interface::Robotiq3FingerStatus_sync sync;
	sync.header = origin->header;
	sync.header.stamp.sec += diff;
	sync.finger_a_status = origin->finger_a_status;
	victor_hardware_interface::Robotiq3FingerStatus_sync::Ptr syncptr(new victor_hardware_interface::Robotiq3FingerStatus_sync(sync));
	// cout << (syncptr->header).stamp << endl;
	return syncptr;
	// return std::make_shared<victor_hardware_interface::Robotiq3FingerStatus_sync> (sync const);
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "create_bagfile");
    ros::NodeHandle nh;

    ros::Publisher left_gripper_pub = nh.advertise<victor_hardware_interface::Robotiq3FingerStatus_sync>("/left_gripper_status", 10);
    ros::Publisher right_gripper_pub = nh.advertise<victor_hardware_interface::Robotiq3FingerStatus_sync>("/right_gripper_status", 10);

    std::vector<std::string> topics;
    topics.push_back(std::string("/kinect2_victor_head/qhd/image_color_rect"));
    topics.push_back(std::string("/kinect2_victor_head/qhd/image_depth_rect"));
    topics.push_back(std::string("/kinect2_victor_head/qhd/camera_info"));
    topics.push_back(std::string("/kinect2_victor_head/qhd/gripper_config"));
    topics.push_back(std::string("/kinect2_victor_head/qhd/dot_config"));
    topics.push_back(std::string("/left_arm/gripper_status_repub"));
    topics.push_back(std::string("/right_arm/gripper_status_repub"));

    rosbag::Bag bag("/home/jingyixiang/cdcpd2_ws/src/cdcpd/cdcpd_ros/dataset/rope_edge_cover_2.bag", rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    bool l_msg_initialized = false;
    bool r_msg_initialized = false;
    for(rosbag::MessageInstance const& m: view) {
        if (m.getTopic() == topics[5] && !l_msg_initialized) {
            auto info = m.instantiate<victor_hardware_interface::Robotiq3FingerStatus>();
            if (info != nullptr) {
                l_msg = gripper_status_origin_to_sync(info, 0);
                // l_sub.newMessage(gripper_status_origin_to_sync(info, 0));
                std::cout << "left gripper msg initialized" << std::endl;
                l_msg_initialized = true;
            } else {
                cout << "NULL initiation!" << endl;
            }
        }
        else if (m.getTopic() == topics[6] && !r_msg_initialized) {
            auto info = m.instantiate<victor_hardware_interface::Robotiq3FingerStatus>();
            if (info != nullptr) {
                r_msg = gripper_status_origin_to_sync(info, 0);
                // r_sub.newMessage(gripper_status_origin_to_sync(info, 0));
                std::cout << "right gripper msg initialized" << std::endl;
                r_msg_initialized = true;
            } else {
                cout << "NULL initiation!" << endl;
            }
        }

        if (l_msg_initialized && r_msg_initialized) {
            break;
        }
    }

    bag.close();

    // test
    while (ros::ok()) {
        l_msg->header.stamp.sec = static_cast<uint32_t>(static_cast<int>(ros::Time::now().toSec()));
        r_msg->header.stamp.sec = static_cast<uint32_t>(static_cast<int>(ros::Time::now().toSec()));
        left_gripper_pub.publish(l_msg);
        right_gripper_pub.publish(r_msg);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    std::cout << "all tasks completed successfully" << std::endl;

    return EXIT_SUCCESS;
}