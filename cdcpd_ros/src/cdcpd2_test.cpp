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

using smmap::AllGrippersSinglePose;
// typedef EigenHelpers::VectorIsometry3d AllGrippersSinglePose;

using smmap::AllGrippersSinglePoseDelta;
// typedef kinematics::VectorVector6d AllGrippersSinglePoseDelta;

using kinematics::Vector6d;
// typedef Eigen::Matrix<double, 6, 1> Vector6d;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
namespace vm = visualization_msgs;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::stringstream;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Matrix3f;
using cv::Matx33d;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::Matrix2Xi;
using Eigen::Matrix2Xf;
using Eigen::Matrix3Xf;
using Eigen::Matrix3Xd;
using Eigen::Isometry3d;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::VectorXi;
using pcl::PointXYZ;

Mat occlusion_mask;
bool updated_opencv_mask = false;
void update_opencv_mask (const sensor_msgs::ImageConstPtr& opencv_mask_msg) {
    occlusion_mask = cv_bridge::toCvShare(opencv_mask_msg, "bgr8")->image;
    if (!occlusion_mask.empty()) {
        updated_opencv_mask = true;
    }
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr Matrix3Xf2pcptr(const Eigen::Matrix3Xf& template_vertices) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < template_vertices.cols(); ++i) 
    {    
        const auto& c = template_vertices.col(i);
        template_cloud->push_back(pcl::PointXYZ(c(0), c(1), c(2)));
    }
	return template_cloud;
}

std::tuple<Eigen::Matrix3Xf, Eigen::Matrix2Xi> init_template()
{
	float left_x = -0.5f; float left_y = 0.2f; float left_z = 0.6f; float right_x = 0.44f; float right_y = 0.2f; float right_z = 0.6f;
    
	int points_on_rope = 30;

    MatrixXf vertices(3, points_on_rope); // Y^0 in the paper
    vertices.setZero();
    vertices.row(0).setLinSpaced(points_on_rope, left_x, right_x);
    vertices.row(1).setLinSpaced(points_on_rope, left_y, right_y);
    vertices.row(2).setLinSpaced(points_on_rope, left_z, right_z);

    MatrixXi edges(2, points_on_rope - 1);
    edges(0, 0) = 0;
    edges(1, edges.cols() - 1) = points_on_rope - 1;
    for (int i = 1; i <= edges.cols() - 1; ++i)
    {
        edges(0, i) = i;
        edges(1, i - 1) = i;
    }

    return std::make_tuple(vertices, edges);
}

// hard coded
std::vector<double> gripper_pt = {0.268377, -0.124886, 0.630706};
std::vector<double> last_gripper_pt = {0.268377, -0.124886, 0.630706};
std::tuple<Eigen::Matrix3Xf, Eigen::Matrix2Xi> init_template_hardcoded()
{
    MatrixXf gmm_Y(35, 3);
    gmm_Y << 0.268377, -0.124886, 0.630706,
                0.247277, -0.124805, 0.632229,
                0.223923, -0.124903, 0.632156,
                0.201773, -0.124571, 0.630938,
                0.183754, -0.122516, 0.630306,
                0.169436, -0.120971, 0.62973,
                0.152277, -0.118236, 0.625117,
                0.138635, -0.114352, 0.623672,
                0.125244, -0.110469, 0.623872,
                0.113181, -0.107071, 0.624976,
                0.0954561, -0.102458, 0.62486,
                0.0768889, -0.0990558, 0.624558,
                0.0630463, -0.0966933, 0.622633,
                0.0464305, -0.0931258, 0.621252,
                0.0299939, -0.0898664, 0.619683,
                0.00994098, -0.0892567, 0.621941,
                -0.00748003, -0.0894387, 0.622552,
                -0.0253184, -0.089625, 0.620576,
                -0.0422711, -0.0919234, 0.621363,
                -0.0554089, -0.0940729, 0.619449,
                -0.0709265, -0.0965604, 0.620223,
                -0.0879523, -0.101123, 0.62019,
                -0.100358, -0.102079, 0.618522,
                -0.115624, -0.104625, 0.618057,
                -0.129764, -0.103671, 0.61819,
                -0.14154, -0.10107, 0.615288,
                -0.153523, -0.0989892, 0.615061,
                -0.167704, -0.0945824, 0.614862,
                -0.17754, -0.0894167, 0.614538,
                -0.1892, -0.0843427, 0.615172,
                -0.201214, -0.0769092, 0.614344,
                -0.214165, -0.0727566, 0.610789,
                -0.226917, -0.0669603, 0.609498,
                -0.239827, -0.0635462, 0.609425,
                -0.250812, -0.0637666, 0.611109;

	int points_on_rope = gmm_Y.rows();

    MatrixXf vertices = gmm_Y.transpose().replicate(1, 1);

    MatrixXi edges(2, points_on_rope - 1);
    edges(0, 0) = 0;
    edges(1, edges.cols() - 1) = points_on_rope - 1;
    for (int i = 1; i <= edges.cols() - 1; ++i)
    {
        edges(0, i) = i;
        edges(1, i - 1) = i;
    }

    std::cout << vertices << std::endl;

    return std::make_tuple(vertices, edges);
}

// initialize cdcpd2
const double alpha = 0.5;
const double lambda = 1.0;
const float zeta = 100.0;
const double k_spring = 100.0;
const double beta = 1.0;
const bool is_sim = false;
const bool is_rope = true;
const bool is_gripper_info = true;	
const double translation_dir_deformability = 1.0;
const double translation_dis_deformability = 1.0;
const double rotation_deformability = 10.0;
// const int points_on_rope = 40;

CDCPD cdcpd;

auto [template_vertices, template_edges] = init_template_hardcoded();
pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud = Matrix3Xf2pcptr(template_vertices);
pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud_init = Matrix3Xf2pcptr(template_vertices);
// end of init

ros::Publisher original_publisher;
ros::Publisher masked_publisher;
ros::Publisher downsampled_publisher;
ros::Publisher pred_publisher;
ros::Publisher output_publisher;

auto frame_id = "camera_color_optical_frame";
bool use_eval_rope = false;
std::shared_ptr<ros::NodeHandle> nh_ptr;

sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::ImageConstPtr& depth_msg) {
    Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask, mask_rgb;
    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    cout << "finished first conversion" << std::endl;
    Mat cur_image_hsv;

    // for cdcpd2
    Mat rgb_image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    cout << "finished second conversion" << std::endl;

    Mat depth_image = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    cout << "finished third conversion" << std::endl;

    // convert color
    cv::cvtColor(cur_image_orig, cur_image_hsv, cv::COLOR_BGR2HSV);

    std::vector<int> lower_blue = {90, 90, 90};
    std::vector<int> upper_blue = {130, 255, 255};

    std::vector<int> lower_red_1 = {130, 60, 40};
    std::vector<int> upper_red_1 = {255, 255, 255};

    std::vector<int> lower_red_2 = {0, 60, 40};
    std::vector<int> upper_red_2 = {10, 255, 255};

    Mat mask_without_occlusion_block;

    if (use_eval_rope) {
        // filter blue
        cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

        // filter red
        cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0], lower_red_1[1], lower_red_1[2]), cv::Scalar(upper_red_1[0], upper_red_1[1], upper_red_1[2]), mask_red_1);
        cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0], lower_red_2[1], lower_red_2[2]), cv::Scalar(upper_red_2[0], upper_red_2[1], upper_red_2[2]), mask_red_2);

        // combine red mask
        cv::bitwise_or(mask_red_1, mask_red_2, mask_red);
        // combine overall mask
        cv::bitwise_or(mask_red, mask_blue, mask_without_occlusion_block);
    }
    else {
        // filter blue
        cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

        mask_blue.copyTo(mask_without_occlusion_block);
    }

    // update cur image for visualization
    Mat cur_image;
    Mat occlusion_mask_gray;
    if (updated_opencv_mask) {
        cv::cvtColor(occlusion_mask, occlusion_mask_gray, cv::COLOR_BGR2GRAY);
        cv::bitwise_and(mask_without_occlusion_block, occlusion_mask_gray, mask);
        cv::bitwise_and(cur_image_orig, occlusion_mask, cur_image);
    }
    else {
        mask_without_occlusion_block.copyTo(mask);
        cur_image_orig.copyTo(cur_image);
    }

    cv::cvtColor(mask, mask_rgb, cv::COLOR_GRAY2BGR);
    // publish mask
    sensor_msgs::ImagePtr mask_msg = nullptr;
    mask_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", mask_rgb).toImageMsg();

    // ========================================================================
    cv::Matx33d placeholder;
    std::vector<CDCPD::FixedPoint> fixed_points;
    
    Eigen::Vector3f left_pos((float)gripper_pt[0], (float)gripper_pt[1], (float)gripper_pt[2]);
    CDCPD::FixedPoint left_gripper = {left_pos, 0};
    Eigen::Vector3f right_pos((float)gripper_pt[0], (float)gripper_pt[1], (float)gripper_pt[2]);
    CDCPD::FixedPoint right_gripper = {right_pos, 0};
    fixed_points.push_back(left_gripper);
    fixed_points.push_back(right_gripper);

    std::vector<bool> is_grasped = {false, true};
    bool is_interaction = false;

    CDCPD::Output out;

    // // ----- no pred -----
    // out = cdcpd(rgb_image, depth_image, mask, placeholder, template_cloud, false, false, false, 0, fixed_points); 
    
    // ----- predictions -----
    // tip node position: -0.259614 -0.00676498     0.61341; probably the left one
    // g_dot has type Eigen::Matrix<double, 6, 1> Vector6d; alias: smmap::AllGrippersSinglePoseDelta
    // g_config has type EigenHelpers::VectorIsometry3d; alias: smmap::AllGrippersSinglePose
    AllGrippersSinglePose one_frame_config;
    AllGrippersSinglePoseDelta one_frame_velocity;

    for (uint32_t g = 0; g < 2; ++g)
    {
        Isometry3d one_config;
        Vector6d one_velocity;

        // for (uint32_t row = 0; row < 4; ++row)
        // {
        //     for (uint32_t col = 0; col < 4; ++col)
        //     {
        //         one_config(row, col) = double(((g_config->data).data)[num_config*g + row*4 + col]);
        //     }
        // }

        one_config(0, 0) = 1.0;
        one_config(0, 1) = 0.0;
        one_config(0, 2) = 0.0;
        one_config(0, 3) = gripper_pt[0];

        one_config(0, 0) = 0.0;
        one_config(0, 1) = 1.0;
        one_config(0, 2) = 0.0;
        one_config(0, 3) = gripper_pt[1];

        one_config(2, 0) = 0.0;
        one_config(2, 1) = 0.0;
        one_config(2, 2) = 1.0;
        one_config(2, 3) = gripper_pt[2];

        one_config(3, 0) = 0.0;
        one_config(3, 1) = 0.0;
        one_config(3, 2) = 0.0;
        one_config(3, 3) = 1.0;

        for (uint32_t i = 0; i < 6; ++i) {
            if (i < 3) {
                one_velocity(i) = gripper_pt[i] - last_gripper_pt[i];
            }
            else {
                one_velocity(i) = 0;
            }
        }

        std::cout << "one_velocity: " << one_velocity << std::endl;

        one_frame_config.push_back(one_config);
        one_frame_velocity.push_back(one_velocity);
    }

    // log time
    std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();

    // // ----- pred 0 -----
    // out = cdcpd(rgb_image, depth_image, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 0, fixed_points);

    // // ----- pred 1 -----
    // out = cdcpd(rgb_image, depth_image, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 1, fixed_points);

    // ----- pred 2 -----
    out = cdcpd(rgb_image, depth_image, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 2, fixed_points);
    
    template_cloud = out.gurobi_output;

    // update gripper point location
    for (int i = 0; i < gripper_pt.size(); i ++) {
        last_gripper_pt[i] = gripper_pt[i];
    }
    gripper_pt = {template_cloud->points[0].x, template_cloud->points[0].y, template_cloud->points[0].z};

    double time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cur_time).count();
    ROS_WARN_STREAM("Total callback time difference: " + std::to_string(time_diff) + " ms");

    // change frame id
    out.gurobi_output->header.frame_id = frame_id;
    out.original_cloud->header.frame_id = frame_id;
    out.masked_point_cloud->header.frame_id = frame_id;
    out.downsampled_cloud->header.frame_id = frame_id;
    out.cpd_predict->header.frame_id = frame_id;

    auto time = ros::Time::now();
    pcl_conversions::toPCL(time, out.original_cloud->header.stamp);
    pcl_conversions::toPCL(time, out.masked_point_cloud->header.stamp);
    pcl_conversions::toPCL(time, out.downsampled_cloud->header.stamp);
    pcl_conversions::toPCL(time, out.gurobi_output->header.stamp);
    // pcl_conversions::toPCL(time, template_cloud_init->header.stamp);
    pcl_conversions::toPCL(time, out.cpd_predict->header.stamp);

    original_publisher.publish(out.original_cloud);
    masked_publisher.publish(out.masked_point_cloud);
    downsampled_publisher.publish(out.downsampled_cloud);
    pred_publisher.publish(out.cpd_predict);
    output_publisher.publish(out.gurobi_output);

    return mask_msg;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "cdcpd2_test");
    ros::NodeHandle nh;

    // cdcpd2 init
    std::vector<float> cylinder_data(8);
    std::vector<float> quat(4);
    // Matrix3Xf init_points(3, 3);

    for (int i = 0; i < 8; i++) {
        cylinder_data[i] = 0.1f;
    }

    for (int i = 0; i < 4; i++) {
        quat[i] = 1.4f;
    }

    cdcpd = CDCPD(template_cloud,
                    template_edges,
                    false,
                    alpha,
                    beta,
                    lambda,
                    k_spring,
                    zeta,
                    cylinder_data,
                    is_sim);
    // end of cdcpd2 init

    nh_ptr = std::make_shared<ros::NodeHandle>(nh);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber opencv_mask_sub = it.subscribe("/mask_with_occlusion", 10, update_opencv_mask);
    image_transport::Publisher mask_pub = it.advertise("/mask", 10);

    original_publisher = nh.advertise<PointCloud> ("cdcpd/original", 1);
    masked_publisher = nh.advertise<PointCloud> ("cdcpd/masked", 1);
    downsampled_publisher = nh.advertise<PointCloud> ("cdcpd/downsampled", 1);
    pred_publisher = nh.advertise<PointCloud>("cdcpd/prediction", 1);
    output_publisher = nh.advertise<PointCloud> ("cdcpd/output", 1);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 10);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_sub, depth_sub, 10);

    sync.registerCallback<std::function<void(const sensor_msgs::ImageConstPtr&, 
                                             const sensor_msgs::ImageConstPtr&,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>)>>
    (
        [&](const sensor_msgs::ImageConstPtr& img_msg, 
            const sensor_msgs::ImageConstPtr& depth_msg,
            const boost::shared_ptr<const message_filters::NullType> var1,
            const boost::shared_ptr<const message_filters::NullType> var2,
            const boost::shared_ptr<const message_filters::NullType> var3,
            const boost::shared_ptr<const message_filters::NullType> var4,
            const boost::shared_ptr<const message_filters::NullType> var5,
            const boost::shared_ptr<const message_filters::NullType> var6,
            const boost::shared_ptr<const message_filters::NullType> var7)
        {
            // sensor_msgs::ImagePtr test_image = imageCallback(msg, _);
            // mask_pub.publish(test_image);
            sensor_msgs::ImagePtr mask_img = Callback(img_msg, depth_msg); // sensor_msgs::ImagePtr tracking_img = 
            mask_pub.publish(mask_img);
        }
    );
    
    ros::spin();
}