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
#include <pcl/filters/voxel_grid.h>
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

// ---------- CONFIG -----------
int num_of_nodes = 41;
bool use_eval_rope = true;
int gripped_idx = 0;

// cdcpd2 params
const bool is_gripper_info = false;

const double alpha = 0.5;
const double lambda = 1.0;
const float zeta = 10.0;
const double k_spring = 100.0;
const double beta = 1.0;
const bool is_sim = false;
const bool is_rope = true;	
const double translation_dir_deformability = 1.0;
const double translation_dis_deformability = 1.0;
const double rotation_deformability = 10.0;
// ---------- END OF CONFIG -----------

template <typename T>
void print_1d_vector (std::vector<T> vec) {
    for (int i = 0; i < vec.size(); i ++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

double pt2pt_dis_sq (MatrixXf pt1, MatrixXf pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

double pt2pt_dis (MatrixXf pt1, MatrixXf pt2) {
    return (pt1 - pt2).rowwise().norm().sum();
}

MatrixXf reg (MatrixXf pts, int M, double mu = 0, int max_iter = 50) {
    // initial guess
    MatrixXf X = pts.replicate(1, 1);
    MatrixXf Y = MatrixXf::Zero(M, 3);
    for (int i = 0; i < M; i ++) {
        Y(i, 1) = 0.1 / static_cast<double>(M) * static_cast<double>(i+1);
        Y(i, 0) = 0;
        Y(i, 2) = 0.63;
    }
    
    int N = X.rows();
    int D = 3;

    // diff_xy should be a (M * N) matrix
    MatrixXf diff_xy = MatrixXf::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    double sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);

    for (int it = 0; it < max_iter; it ++) {
        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        MatrixXf P = (-0.5 * diff_xy / sigma2).array().exp();
        MatrixXf P_stored = P.replicate(1, 1);
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        MatrixXf Pt1 = P.colwise().sum(); 
        MatrixXf P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXf PX = P * X;

        MatrixXf P1_expanded = MatrixXf::Zero(M, D);
        P1_expanded.col(0) = P1;
        P1_expanded.col(1) = P1;
        P1_expanded.col(2) = P1;

        Y = PX.cwiseQuotient(P1_expanded);

        double numerator = 0;
        double denominator = 0;

        for (int m = 0; m < M; m ++) {
            for (int n = 0; n < N; n ++) {
                numerator += P(m, n)*diff_xy(m, n);
                denominator += P(m, n)*D;
            }
        }

        sigma2 = numerator / denominator;
    }

    return Y;
}

MatrixXf sort_pts (MatrixXf Y_0) {
    int N = Y_0.rows();
    MatrixXf Y_0_sorted = MatrixXf::Zero(N, 3);
    std::vector<MatrixXf> Y_0_sorted_vec = {};
    std::vector<bool> selected_node(N, false);
    selected_node[0] = true;
    int last_visited_b = 0;

    MatrixXf G = MatrixXf::Zero(N, N);
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            G(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
        }
    }

    int reverse = 0;
    int counter = 0;
    int reverse_on = 0;
    int insertion_counter = 0;

    while (counter < N-1) {
        double minimum = INFINITY;
        int a = 0;
        int b = 0;

        for (int m = 0; m < N; m ++) {
            if (selected_node[m] == true) {
                for (int n = 0; n < N; n ++) {
                    if ((!selected_node[n]) && (G(m, n) != 0.0)) {
                        if (minimum > G(m, n)) {
                            minimum = G(m, n);
                            a = m;
                            b = n;
                        }
                    }
                }
            }
        }

        if (counter == 0) {
            Y_0_sorted_vec.push_back(Y_0.row(a));
            Y_0_sorted_vec.push_back(Y_0.row(b));
        }
        else {
            if (last_visited_b != a) {
                reverse += 1;
                reverse_on = a;
                insertion_counter = 1;
            }
            
            if (reverse % 2 == 1) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(a));
                Y_0_sorted_vec.insert(it, Y_0.row(b));
            }
            else if (reverse != 0) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(reverse_on));
                Y_0_sorted_vec.insert(it + insertion_counter, Y_0.row(b));
                insertion_counter += 1;
            }
            else {
                Y_0_sorted_vec.push_back(Y_0.row(b));
            }
        }

        last_visited_b = b;
        selected_node[b] = true;
        counter += 1;
    }

    // copy to Y_0_sorted
    for (int i = 0; i < N; i ++) {
        Y_0_sorted.row(i) = Y_0_sorted_vec[i];
    }

    return Y_0_sorted;
}

ros::Publisher results_pub;

// node color and object color are in rgba format and range from 0-1
visualization_msgs::MarkerArray MatrixXf2MarkerArray (MatrixXf gripped_pt, MatrixXf Y, std::string marker_frame, std::string marker_ns, std::vector<float> node_color, std::vector<float> line_color) {
    // publish the results as a marker array
    visualization_msgs::MarkerArray results = visualization_msgs::MarkerArray();

    visualization_msgs::Marker cur_node_result = visualization_msgs::Marker();
    
    // add header
    cur_node_result.header.frame_id = marker_frame;
    // cur_node_result.header.stamp = ros::Time::now();
    cur_node_result.type = visualization_msgs::Marker::SPHERE;
    cur_node_result.action = visualization_msgs::Marker::ADD;
    cur_node_result.ns = marker_ns + std::to_string(99);
    cur_node_result.id = 99;

    // add position
    cur_node_result.pose.position.x = gripped_pt(0, 0);
    cur_node_result.pose.position.y = gripped_pt(0, 1);
    cur_node_result.pose.position.z = gripped_pt(0, 2);

    // add orientation
    cur_node_result.pose.orientation.w = 1.0;
    cur_node_result.pose.orientation.x = 0.0;
    cur_node_result.pose.orientation.y = 0.0;
    cur_node_result.pose.orientation.z = 0.0;

    // set scale
    cur_node_result.scale.x = 0.01;
    cur_node_result.scale.y = 0.01;
    cur_node_result.scale.z = 0.01;

    // set color
    cur_node_result.color.r = 1.0;
    cur_node_result.color.g = 0.0;
    cur_node_result.color.b = 0.0;
    cur_node_result.color.a = 1.0;

    results.markers.push_back(cur_node_result);

    for (int i = 0; i < Y.rows(); i ++) {
        visualization_msgs::Marker cur_node_result = visualization_msgs::Marker();
    
        // add header
        cur_node_result.header.frame_id = marker_frame;
        // cur_node_result.header.stamp = ros::Time::now();
        cur_node_result.type = visualization_msgs::Marker::SPHERE;
        cur_node_result.action = visualization_msgs::Marker::ADD;
        cur_node_result.ns = marker_ns + std::to_string(i);
        cur_node_result.id = i;

        // add position
        cur_node_result.pose.position.x = Y(i, 0);
        cur_node_result.pose.position.y = Y(i, 1);
        cur_node_result.pose.position.z = Y(i, 2);

        // add orientation
        cur_node_result.pose.orientation.w = 1.0;
        cur_node_result.pose.orientation.x = 0.0;
        cur_node_result.pose.orientation.y = 0.0;
        cur_node_result.pose.orientation.z = 0.0;

        // set scale
        cur_node_result.scale.x = 0.01;
        cur_node_result.scale.y = 0.01;
        cur_node_result.scale.z = 0.01;

        // set color
        cur_node_result.color.r = node_color[0];
        cur_node_result.color.g = node_color[1];
        cur_node_result.color.b = node_color[2];
        cur_node_result.color.a = node_color[3];

        results.markers.push_back(cur_node_result);

        // don't add line if at the last node
        if (i == Y.rows()-1) {
            break;
        }

        visualization_msgs::Marker cur_line_result = visualization_msgs::Marker();

        // add header
        cur_line_result.header.frame_id = "camera_color_optical_frame";
        cur_line_result.type = visualization_msgs::Marker::CYLINDER;
        cur_line_result.action = visualization_msgs::Marker::ADD;
        cur_line_result.ns = "line_results" + std::to_string(i);
        cur_line_result.id = i;

        // add position
        cur_line_result.pose.position.x = (Y(i, 0) + Y(i+1, 0)) / 2.0;
        cur_line_result.pose.position.y = (Y(i, 1) + Y(i+1, 1)) / 2.0;
        cur_line_result.pose.position.z = (Y(i, 2) + Y(i+1, 2)) / 2.0;

        // add orientation
        Eigen::Quaternionf q;
        Eigen::Vector3f vec1(0.0, 0.0, 1.0);
        Eigen::Vector3f vec2(Y(i+1, 0) - Y(i, 0), Y(i+1, 1) - Y(i, 1), Y(i+1, 2) - Y(i, 2));
        q.setFromTwoVectors(vec1, vec2);

        cur_line_result.pose.orientation.w = q.w();
        cur_line_result.pose.orientation.x = q.x();
        cur_line_result.pose.orientation.y = q.y();
        cur_line_result.pose.orientation.z = q.z();

        // set scale
        cur_line_result.scale.x = 0.005;
        cur_line_result.scale.y = 0.005;
        cur_line_result.scale.z = pt2pt_dis(Y.row(i), Y.row(i+1));

        // set color
        cur_line_result.color.r = line_color[0];
        cur_line_result.color.g = line_color[1];
        cur_line_result.color.b = line_color[2];
        cur_line_result.color.a = line_color[3];

        results.markers.push_back(cur_line_result);
    }

    return results;
}

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
	float left_x = -0.2f; float left_y = 0.0f; float left_z = 0.6f; float right_x = 0.6f; float right_y = 0.0f; float right_z = 0.6f;
    
	int points_on_rope = num_of_nodes;

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

std::vector<double> gripper_pt;
std::vector<double> last_gripper_pt;

std::tuple<Eigen::Matrix3Xf, Eigen::Matrix2Xi> init_template_gmm(MatrixXf Y_gmm)
{
	int points_on_rope = Y_gmm.rows();

    MatrixXf vertices = Y_gmm.transpose().replicate(1, 1);

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

CDCPD cdcpd;

// auto [template_vertices, template_edges] = init_template_hardcoded();
// pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud = Matrix3Xf2pcptr(template_vertices);
// pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud_init = Matrix3Xf2pcptr(template_vertices);
Eigen::Matrix3Xf template_vertices;
Eigen::Matrix2Xi template_edges;
pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud;
pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud_init;
// end of init

ros::Publisher original_publisher;
ros::Publisher masked_publisher;
ros::Publisher downsampled_publisher;
ros::Publisher pred_publisher;
ros::Publisher output_publisher;

auto frame_id = "camera_color_optical_frame";
std::shared_ptr<ros::NodeHandle> nh_ptr;

// ---------- CALLBACK ----------
bool initialized = false;
std::chrono::steady_clock::time_point gripper_time = std::chrono::steady_clock::now();

// sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
    Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask_yellow, mask, mask_rgb;
    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    cout << "finished first conversion" << std::endl;
    Mat cur_image_hsv;

    // for cdcpd2
    Mat rgb_image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    cout << "finished second conversion" << std::endl;

    // Mat depth_image = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    // cout << "finished third conversion" << std::endl;

    // convert color
    cv::cvtColor(cur_image_orig, cur_image_hsv, cv::COLOR_BGR2HSV);

    std::vector<int> lower_blue = {90, 90, 90};
    std::vector<int> upper_blue = {130, 255, 255};

    std::vector<int> lower_red_1 = {130, 60, 40};
    std::vector<int> upper_red_1 = {255, 255, 255};

    std::vector<int> lower_red_2 = {0, 60, 40};
    std::vector<int> upper_red_2 = {10, 255, 255};

    std::vector<int> lower_yellow = {15, 100, 80};
    std::vector<int> upper_yellow = {40, 255, 255};

    Mat mask_without_occlusion_block;

    if (use_eval_rope) {
        // filter blue
        cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

        // filter red
        cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0], lower_red_1[1], lower_red_1[2]), cv::Scalar(upper_red_1[0], upper_red_1[1], upper_red_1[2]), mask_red_1);
        cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0], lower_red_2[1], lower_red_2[2]), cv::Scalar(upper_red_2[0], upper_red_2[1], upper_red_2[2]), mask_red_2);

        // combine red mask
        cv::bitwise_or(mask_red_1, mask_red_2, mask_red);

        // filter yellow
        cv::inRange(cur_image_hsv, cv::Scalar(lower_yellow[0], lower_yellow[1], lower_yellow[2]), cv::Scalar(upper_yellow[0], upper_yellow[1], upper_yellow[2]), mask_yellow);

        // combine overall mask
        cv::bitwise_or(mask_red, mask_blue, mask_without_occlusion_block);
        cv::bitwise_or(mask_yellow, mask_without_occlusion_block, mask_without_occlusion_block);
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

    // deal with point cloud 
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
    // Convert to PCL data type
    pcl_conversions::toPCL(*pc_msg, *cloud);   // cloud is 720*1280 (height*width) now, however is a ros pointcloud2 message. 
                                            // see message definition here: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html

    if (cloud->width == 0 && cloud->height == 0) {
        ROS_ERROR("empty point cloud!");
        return mask_msg;
    }

    std::cout << "non empty point cloud" << std::endl;

    // convert to xyz point
    pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz;
    pcl::fromPCLPointCloud2(*cloud, cloud_xyz);
    // now create objects for cur_pc
    pcl::PCLPointCloud2* cur_pc = new pcl::PCLPointCloud2;
    pcl::PointCloud<pcl::PointXYZRGB> cur_pc_xyz;
    pcl::PointCloud<pcl::PointXYZRGB> cur_nodes_xyz;
    pcl::PointCloud<pcl::PointXYZRGB> downsampled_xyz;

    pcl::PointCloud<pcl::PointXYZRGB> cur_yellow_xyz;

    // temp depth from pc
    Mat depth_image = Mat::zeros(1280, 720, CV_64F);

    // filter point cloud from mask
    for (int i = 0; i < cloud->height; i ++) {
        for (int j = 0; j < cloud->width; j ++) {
            depth_image.at<uchar>(i, j) = cloud_xyz(j, i).z;
            if (mask.at<uchar>(i, j) != 0) {
                cur_pc_xyz.push_back(cloud_xyz(j, i));   // note: this is (j, i) not (i, j)
            }
            if (mask_yellow.at<uchar>(i, j) != 0) {
                cur_yellow_xyz.push_back(cloud_xyz(j, i));   // note: this is (j, i) not (i, j)
            }
        }
    }

    MatrixXf cur_yellow_pts = cur_yellow_xyz.getMatrixXfMap().topRows(3).transpose();
    std::cout << cur_yellow_pts.rows() << std::endl;
    MatrixXf head_node = reg(cur_yellow_pts, 1, 0.1, 100);
    std::cout << "head node: " << head_node << std::endl;

    // convert back to pointcloud2 message
    pcl::toPCLPointCloud2(cur_pc_xyz, *cur_pc);
    // Perform downsampling
    pcl::PCLPointCloud2ConstPtr cloudPtr(cur_pc);
    pcl::PCLPointCloud2 cur_pc_downsampled;
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloudPtr);
    sor.setLeafSize (0.005, 0.005, 0.005);
    sor.filter (cur_pc_downsampled);

    pcl::fromPCLPointCloud2(cur_pc_downsampled, downsampled_xyz);

    if (!initialized) {
        MatrixXf X = downsampled_xyz.getMatrixXfMap().topRows(3).transpose();

        std::cout << "found X" << std::endl;

        MatrixXf Y_gmm = reg(X, num_of_nodes, 0.05, 100);
        Y_gmm = sort_pts(Y_gmm);

        std::cout << "found Y" << std::endl;

        gripper_pt = {head_node(0, 0), head_node(0, 1), head_node(0, 2)};
        last_gripper_pt = {head_node(0, 0), head_node(0, 1), head_node(0, 2)};

        // auto [template_vertices_, template_edges_] = init_template_gmm(Y_gmm);
        auto [template_vertices_, template_edges_] = init_template();
        template_vertices = template_vertices_.replicate(1, 1);
        template_edges = template_edges_.replicate(1, 1);
        template_cloud = Matrix3Xf2pcptr(template_vertices);
        template_cloud_init = Matrix3Xf2pcptr(template_vertices);

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

        initialized = true;
    }

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
    // out = cdcpd(rgb_image, depth_image, pc_msg, mask, placeholder, template_cloud, false, false, false, 0, fixed_points); 
    
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

        one_config(1, 0) = 0.0;
        one_config(1, 1) = 1.0;
        one_config(1, 2) = 0.0;
        one_config(1, 3) = gripper_pt[1];

        one_config(2, 0) = 0.0;
        one_config(2, 1) = 0.0;
        one_config(2, 2) = 1.0;
        one_config(2, 3) = gripper_pt[2];

        one_config(3, 0) = 0.0;
        one_config(3, 1) = 0.0;
        one_config(3, 2) = 0.0;
        one_config(3, 3) = 1.0;

        // log time
        double gripper_time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gripper_time).count();

        for (uint32_t i = 0; i < 6; ++i) {
            if (i < 3) {
                one_velocity(i) = (gripper_pt[i] - last_gripper_pt[i]); // / (gripper_time_diff / 1000.0);
                if (one_velocity(i) < 1e-4) {
                    one_velocity(i) = 0;
                }
                // else {
                //     one_velocity(i) = (gripper_pt[i] - last_gripper_pt[i]) / (gripper_time_diff / 1000.0);
                // }
                
                // std::cout << gripper_time_diff/1000.0 << std::endl;
                std::cout << "orig vel = " << gripper_pt[i] - last_gripper_pt[i] << std::endl;
                std::cout << "scaled vel = " << (gripper_pt[i] - last_gripper_pt[i]) / (gripper_time_diff / 1000.0) << std::endl;
            }
            else {
                one_velocity(i) = 0;
            }
        }

        std::cout << "one_velocity: " << one_velocity << std::endl;

        one_frame_config.push_back(one_config);
        one_frame_velocity.push_back(one_velocity);
    }

    // update gripper point location
    for (int i = 0; i < gripper_pt.size(); i ++) {
        last_gripper_pt[i] = gripper_pt[i];
    }
    gripper_pt = {head_node(0, 0), head_node(0, 1), head_node(0, 2)};

    // log time
    gripper_time = std::chrono::steady_clock::now();

    // log time
    std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();

    if (is_gripper_info) {
        // pred_choice:
        // 	- 0: no movement
        // 	- 1: Dmitry's prediction
        //  - 2: Mengyao's prediction

        // // ----- pred 0 -----
        // out = cdcpd(rgb_image, depth_image, pc_msg, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 0, fixed_points);

        // // ----- pred 1 -----
        // out = cdcpd(rgb_image, depth_image, pc_msg, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 1, fixed_points);

        // ----- pred 2 -----
        out = cdcpd(rgb_image, depth_image, downsampled_xyz, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 2, fixed_points);
    }
    else {
        std::vector<CDCPD::FixedPoint> fixed_points = {};
        out = cdcpd(rgb_image, depth_image, downsampled_xyz, mask, placeholder, template_cloud, false, false, false, 0, fixed_points);
    }

    
    template_cloud = out.gurobi_output;

    // convert to MatrixXf
    MatrixXf Y = MatrixXf::Zero(num_of_nodes, 3);
    for (int i = 0; i < num_of_nodes; i ++) {
        Y(i, 0) = template_cloud->points[i].x;
        Y(i, 1) = template_cloud->points[i].y;
        Y(i, 2) = template_cloud->points[i].z;
    }
    // publish marker array
    visualization_msgs::MarkerArray results = MatrixXf2MarkerArray(head_node, Y, "camera_color_optical_frame", "node_results", {1.0, 150.0/255.0, 0.0, 0.75}, {0.0, 1.0, 0.0, 0.75});
    results_pub.publish(results);

    double time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cur_time).count();
    ROS_WARN_STREAM("Total callback time difference: " + std::to_string(time_diff) + " ms");

    // change frame id
    out.gurobi_output->header.frame_id = frame_id;
    out.original_cloud.header.frame_id = frame_id;
    out.masked_point_cloud.header.frame_id = frame_id;
    out.downsampled_cloud.header.frame_id = frame_id;
    out.cpd_predict->header.frame_id = frame_id;

    auto time = ros::Time::now();
    pcl_conversions::toPCL(time, out.original_cloud.header.stamp);
    pcl_conversions::toPCL(time, out.masked_point_cloud.header.stamp);
    pcl_conversions::toPCL(time, out.downsampled_cloud.header.stamp);
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

    nh_ptr = std::make_shared<ros::NodeHandle>(nh);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber opencv_mask_sub = it.subscribe("/mask_with_occlusion", 10, update_opencv_mask);
    image_transport::Publisher mask_pub = it.advertise("/mask", 10);
    results_pub = nh.advertise<visualization_msgs::MarkerArray>("/results_marker", 1);

    original_publisher = nh.advertise<PointCloud> ("cdcpd/original", 1);
    masked_publisher = nh.advertise<PointCloud> ("cdcpd/masked", 1);
    downsampled_publisher = nh.advertise<PointCloud> ("cdcpd/downsampled", 1);
    pred_publisher = nh.advertise<PointCloud>("cdcpd/prediction", 1);
    output_publisher = nh.advertise<PointCloud> ("cdcpd/output", 1);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/camera/depth/color/points", 10);
    // message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> sync(image_sub, depth_sub, pc_sub, 10);

    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync(image_sub, pc_sub, 10);

    sync.registerCallback<std::function<void(const sensor_msgs::ImageConstPtr&, 
                                             const sensor_msgs::PointCloud2ConstPtr&,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>,
                                             const boost::shared_ptr<const message_filters::NullType>)>>
    (
        [&](const sensor_msgs::ImageConstPtr& img_msg, 
            const sensor_msgs::PointCloud2ConstPtr& pc_msg,
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
            // sensor_msgs::ImagePtr mask_img = Callback(img_msg, depth_msg, pc_msg); // sensor_msgs::ImagePtr tracking_img = 
            sensor_msgs::ImagePtr mask_img = Callback(img_msg, pc_msg);
            mask_pub.publish(mask_img);
        }
    );
    
    ros::spin();
}