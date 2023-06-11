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
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

#include <cdcpd_ros/Float32MultiArrayStamped.h>
#include <victor_hardware_interface_msgs/Robotiq3FingerStatus_sync.h>
#include <victor_hardware_interface_msgs/Robotiq3FingerStatus.h>

#include <Eigen/Geometry>

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
using Eigen::MatrixXf;
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

ros::Subscriber camera_info_sub;

MatrixXf proj_matrix(3, 4);
bool received_proj_matrix = false;
ros::Publisher results_pub;

// read gripper tf
tf2_ros::Buffer tfBuffer;

Mat occlusion_mask;
bool updated_opencv_mask = false;
std::vector<double> gripper_pt;
std::vector<double> last_gripper_pt;

// ---------- CONFIG -----------
int num_of_nodes;

// cdcpd2 params
bool is_gripper_info;
bool use_real_gripper;

// 0 -> statinary.bag; 1 -> with_gripper_perpendicular.bag
int bag_file;
double bag_rate;

bool multi_color_dlo;
double alpha;
double beta;
double lambda;
float zeta;
double downsample_leaf_size;

double k_spring;
bool is_sim;
bool is_rope;	
double translation_dir_deformability;
double translation_dis_deformability;
double rotation_deformability;

// rope pos initialization
double left_x; 
double left_y; 
double left_z; 
double right_x; 
double right_y; 
double right_z;

// color thresholding 
std::string camera_info_topic;
std::string rgb_topic;
std::string depth_topic;
std::string hsv_threshold_upper_limit;
std::string hsv_threshold_lower_limit;
std::vector<int> upper;
std::vector<int> lower;
double visibility_threshold;

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

void update_camera_info (const sensor_msgs::CameraInfoConstPtr& cam_msg) {
    auto P = cam_msg->P;
    for (int i = 0; i < P.size(); i ++) {
        proj_matrix(i/4, i%4) = P[i];
    }
    std::cout << "received projection matrix" << std::endl;
    received_proj_matrix = true;
    camera_info_sub.shutdown();
}

// node color and object color are in rgba format and range from 0-1
visualization_msgs::MarkerArray MatrixXf2MarkerArray (MatrixXf Y, std::string marker_frame, std::string marker_ns, std::vector<float> node_color, std::vector<float> line_color) {
    // publish the results as a marker array
    visualization_msgs::MarkerArray results = visualization_msgs::MarkerArray();

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

    // std::cout << vertices << std::endl;

    return std::make_tuple(vertices, edges);
}

Mat color_thresholding (Mat cur_image_hsv) {
    std::vector<int> lower_blue = {90, 90, 60};
    std::vector<int> upper_blue = {130, 255, 255};

    std::vector<int> lower_red_1 = {130, 60, 50};
    std::vector<int> upper_red_1 = {255, 255, 255};

    std::vector<int> lower_red_2 = {0, 60, 50};
    std::vector<int> upper_red_2 = {10, 255, 255};

    std::vector<int> lower_yellow = {15, 100, 80};
    std::vector<int> upper_yellow = {40, 255, 255};

    Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask_yellow, mask;
    // filter blue
    cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0], lower_blue[1], lower_blue[2]), cv::Scalar(upper_blue[0], upper_blue[1], upper_blue[2]), mask_blue);

    // filter red
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0], lower_red_1[1], lower_red_1[2]), cv::Scalar(upper_red_1[0], upper_red_1[1], upper_red_1[2]), mask_red_1);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0], lower_red_2[1], lower_red_2[2]), cv::Scalar(upper_red_2[0], upper_red_2[1], upper_red_2[2]), mask_red_2);

    // filter yellow
    cv::inRange(cur_image_hsv, cv::Scalar(lower_yellow[0], lower_yellow[1], lower_yellow[2]), cv::Scalar(upper_yellow[0], upper_yellow[1], upper_yellow[2]), mask_yellow);

    // combine red mask
    cv::bitwise_or(mask_red_1, mask_red_2, mask_red);
    // combine overall mask
    cv::bitwise_or(mask_red, mask_blue, mask);
    cv::bitwise_or(mask_yellow, mask, mask);

    return mask;
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

std::chrono::steady_clock::time_point start_time;

// ---------- CALLBACK ----------
bool initialized = false;
bool reversed_Y = false;
MatrixXf Y_init;

std::chrono::steady_clock::time_point gripper_time = std::chrono::steady_clock::now();

// sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
sensor_msgs::ImagePtr Callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::ImageConstPtr& depth_msg) {
    
    Mat cur_image_orig = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    Mat cur_depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;

    // will get overwritten later if intialized
    sensor_msgs::ImagePtr tracking_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur_image_orig).toImageMsg();

    std::cout << "finished image conversions" << std::endl;

    if (received_proj_matrix) {
        if (!initialized) {
            start_time = std::chrono::steady_clock::now();
        }
        // log time
        std::chrono::steady_clock::time_point cur_time_cb = std::chrono::steady_clock::now();

        Mat mask, mask_rgb, mask_without_occlusion_block;
        Mat cur_image_hsv;

        // convert color
        cv::cvtColor(cur_image_orig, cur_image_hsv, cv::COLOR_BGR2HSV);

        if (!multi_color_dlo) {
            // color_thresholding
            cv::inRange(cur_image_hsv, cv::Scalar(lower[0], lower[1], lower[2]), cv::Scalar(upper[0], upper[1], upper[2]), mask_without_occlusion_block);
        }
        else {
            mask_without_occlusion_block = color_thresholding(cur_image_hsv);
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

        // initialize point cloud for publishing results
        sensor_msgs::PointCloud2 cur_pc_downsampled_pointcloud2;
        sensor_msgs::PointCloud2 result_pc;

        bool simulated_occlusion = false;
        int occlusion_corner_i = -1;
        int occlusion_corner_j = -1;
        int occlusion_corner_i_2 = -1;
        int occlusion_corner_j_2 = -1;

        // filter point cloud
        pcl::PointCloud<pcl::PointXYZRGB> cur_pc;
        pcl::PointCloud<pcl::PointXYZRGB> cur_pc_downsampled;

        // filter point cloud from mask
        for (int i = 0; i < mask.rows; i ++) {
            for (int j = 0; j < mask.cols; j ++) {
                // for text label (visualization)
                if (updated_opencv_mask && !simulated_occlusion && occlusion_mask_gray.at<uchar>(i, j) == 0) {
                    occlusion_corner_i = i;
                    occlusion_corner_j = j;
                    simulated_occlusion = true;
                }

                // update the other corner of occlusion mask (visualization)
                if (updated_opencv_mask && occlusion_mask_gray.at<uchar>(i, j) == 0) {
                    occlusion_corner_i_2 = i;
                    occlusion_corner_j_2 = j;
                }

                double depth_threshold = 0.4 * 1000;  // millimeters
                if (mask.at<uchar>(i, j) != 0 && cur_depth.at<uint16_t>(i, j) > depth_threshold) {
                    // point cloud from image pixel coordinates and depth value
                    pcl::PointXYZRGB point;
                    double pixel_x = static_cast<double>(j);
                    double pixel_y = static_cast<double>(i);
                    double cx = proj_matrix(0, 2);
                    double cy = proj_matrix(1, 2);
                    double fx = proj_matrix(0, 0);
                    double fy = proj_matrix(1, 1);
                    double pc_z = cur_depth.at<uint16_t>(i, j) / 1000.0;

                    point.x = (pixel_x - cx) * pc_z / fx;
                    point.y = (pixel_y - cy) * pc_z / fy;
                    point.z = pc_z;

                    // currently missing point field so color doesn't show up in rviz
                    point.r = cur_image_orig.at<cv::Vec3b>(i, j)[0];
                    point.g = cur_image_orig.at<cv::Vec3b>(i, j)[1];
                    point.b = cur_image_orig.at<cv::Vec3b>(i, j)[2];

                    cur_pc.push_back(point);
                }
            }
        }

        // Perform downsampling
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloudPtr(cur_pc.makeShared());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (cloudPtr);
        sor.setLeafSize (downsample_leaf_size, downsample_leaf_size, downsample_leaf_size);
        sor.filter(cur_pc_downsampled);

        geometry_msgs::TransformStamped gripper_tf;
        if (use_real_gripper) {
            gripper_tf = tfBuffer.lookupTransform("camera_color_optical_frame", "tool0_m", ros::Time(0));
            std::cout << "translation: " << gripper_tf.transform.translation.x << "; " << gripper_tf.transform.translation.y << "; " << gripper_tf.transform.translation.z << std::endl;
        }

        std::cout << "finished point cloud processing" << std::endl;

        if (!initialized) {
            // MatrixXf X = cur_pc_downsampled.getMatrixXfMap().topRows(3).transpose();
            // MatrixXf Y_gmm = reg(X, num_of_nodes, 0.05, 100);
            // Y_gmm = sort_pts(Y_gmm);

            std::cout << "uninitialized" << std::endl;

            if (is_gripper_info && use_real_gripper) {
                gripper_pt = {gripper_tf.transform.translation.x, gripper_tf.transform.translation.y, gripper_tf.transform.translation.z};
                last_gripper_pt = {gripper_tf.transform.translation.x, gripper_tf.transform.translation.y, gripper_tf.transform.translation.z};
            }
            else {
                gripper_pt = {0.0, 0.0, 0.0};
                last_gripper_pt = {0.0, 0.0, 0.0};
            }

            // auto [template_vertices_, template_edges_] = init_template_gmm(Y_gmm);
            auto [template_vertices_, template_edges_] = init_template();
            // auto [template_vertices_, template_edges_] = init_template_hardcoded();
            template_vertices = template_vertices_.replicate(1, 1);
            template_edges = template_edges_.replicate(1, 1);
            template_cloud = Matrix3Xf2pcptr(template_vertices);
            template_cloud_init = Matrix3Xf2pcptr(template_vertices);

            std::cout << "finished template initializaiton" << std::endl;

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
            cdcpd.kvis = 100;
            // end of cdcpd2 init

            // Eigen::Matrix3f intrinsics_eigen(3, 3);
            // intrinsics_eigen << 918.359130859375, 0.0, 645.8908081054688,
            //                     0.0, 916.265869140625, 354.02392578125,
            //                     0.0, 0.0, 1.0;
            // intrinsics_eigen.cast<float>();

            // for (int i = 0; i < 10; i ++) {
            //     std::cout << "running initializaiton cpd" << std::endl;
            //     cdcpd.cpd(X, template_vertices, Y_init.transpose(), cur_depth, mask, intrinsics_eigen);
            // }
            
            // template_cloud = Matrix3Xf2pcptr(template_vertices);

            initialized = true;

            std::cout << "finished cdcpd2 initialization" << std::endl;
        }
        else {
            // ========================================================================
            cv::Matx33d placeholder;
            std::vector<CDCPD::FixedPoint> fixed_points;
            Eigen::Vector3f left_pos;
            CDCPD::FixedPoint left_gripper;
            Eigen::Vector3f right_pos;
            CDCPD::FixedPoint right_gripper;
            std::vector<bool> is_grasped;

            if (is_gripper_info && use_real_gripper) {
                gripper_pt = {gripper_tf.transform.translation.x, gripper_tf.transform.translation.y, gripper_tf.transform.translation.z};
                left_pos << (float)gripper_pt[0], (float)gripper_pt[1], (float)gripper_pt[2];
                left_gripper = {left_pos, 0};

                right_pos << (float)gripper_pt[0], (float)gripper_pt[1], (float)gripper_pt[2];
                right_gripper = {right_pos, 0};

                fixed_points.push_back(left_gripper);
                fixed_points.push_back(right_gripper);

                is_grasped = {false, true};
            }
            else {
                left_pos << 0.0, 0.0, 0.0;
                left_gripper = {left_pos, 0};

                right_pos << 0.0, 0.0, 0.0;
                right_gripper = {right_pos, 0};

                fixed_points.push_back(left_gripper);
                fixed_points.push_back(right_gripper);
                
                is_grasped = {false, false};
            }
            // std::cout << right_pos << std::endl;
            bool is_interaction = false;

            std::cout << "finished gripper point initialization" << std::endl;

            CDCPD::Output out;

            // // ----- no pred -----
            // out = cdcpd(rgb_image, cur_depth, pc_msg, mask, placeholder, template_cloud, false, false, false, 0, fixed_points); 
            
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

                std::cout << "in for loop" << std::endl;

                if (!use_real_gripper || !is_gripper_info) {
                    std::cout << "before modifying one_config" << std::endl;
                    one_config(0, 0) = 1.0;
                    one_config(0, 1) = 0.0;
                    one_config(0, 2) = 0.0;

                    one_config(1, 0) = 0.0;
                    one_config(1, 1) = 1.0;
                    one_config(1, 2) = 0.0;

                    one_config(2, 0) = 0.0;
                    one_config(2, 1) = 0.0;
                    one_config(2, 2) = 1.0;
                }
                else {
                    Eigen::Quaterniond q;
                    q.w() = gripper_tf.transform.rotation.w;
                    q.x() = gripper_tf.transform.rotation.x;
                    q.y() = gripper_tf.transform.rotation.y;
                    q.z() = gripper_tf.transform.rotation.z;

                    auto R = q.normalized().toRotationMatrix();

                    one_config(0, 0) = R(0, 0);
                    one_config(0, 1) = R(0, 1);
                    one_config(0, 2) = R(0, 2);

                    one_config(1, 0) = R(1, 0);
                    one_config(1, 1) = R(1, 1);
                    one_config(1, 2) = R(1, 2);

                    one_config(2, 0) = R(2, 0);
                    one_config(2, 1) = R(2, 1);
                    one_config(2, 2) = R(2, 2);
                }

                one_config(3, 0) = 0.0;
                one_config(3, 1) = 0.0;
                one_config(3, 2) = 0.0;
                one_config(3, 3) = 1.0;

                one_config(0, 3) = gripper_pt[0];
                one_config(1, 3) = gripper_pt[1];
                one_config(2, 3) = gripper_pt[2];
                // if (g == 1) {
                //     one_config(0, 3) = gripper_pt[0];
                //     one_config(1, 3) = gripper_pt[1];
                //     one_config(2, 3) = gripper_pt[2];
                // }
                // else {
                //     one_config(0, 3) = Y_init(Y_init.rows()-1, 0);
                //     one_config(1, 3) = Y_init(Y_init.rows()-1, 1);
                //     one_config(2, 3) = Y_init(Y_init.rows()-1, 2);
                // }

                std::cout << "finished first for loop" << std::endl;

                // log time
                double gripper_time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gripper_time).count();

                for (uint32_t i = 0; i < 6; ++i) {
                    if (is_gripper_info) {
                        if (i < 3) {
                            one_velocity(i) = (gripper_pt[i] - last_gripper_pt[i]); // / (gripper_time_diff / 1000.0);
                            if (one_velocity(i) < 1e-4) {
                                one_velocity(i) = 0;
                            }
                            // else {
                            //     one_velocity(i) = (gripper_pt[i] - last_gripper_pt[i]) / (gripper_time_diff / 1000.0);
                            // }
                            
                            // std::cout << gripper_time_diff/1000.0 << std::endl;
                            // std::cout << "vel = " << gripper_pt[i] - last_gripper_pt[i] << std::endl;
                            // std::cout << "scaled vel = " << (gripper_pt[i] - last_gripper_pt[i]) / (gripper_time_diff / 1000.0) << std::endl;
                        }
                        else {
                            one_velocity(i) = 0;
                        }
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

            // log time
            gripper_time = std::chrono::steady_clock::now();

            // log time
            std::chrono::steady_clock::time_point cur_time = std::chrono::steady_clock::now();

            std::cout << "before registration" << std::endl;

            if (is_gripper_info) {
                // pred_choice:
                // 	- 0: no movement
                // 	- 1: Dmitry's prediction
                //  - 2: Mengyao's prediction

                // // ----- pred 0 -----
                // out = cdcpd(rgb_image, cur_depth, pc_msg, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 0, fixed_points);

                // // ----- pred 1 -----
                // out = cdcpd(rgb_image, cur_depth, pc_msg, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 1, fixed_points);

                // ----- pred 2 -----
                out = cdcpd(cur_image_orig, cur_depth, cur_pc_downsampled, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 2, fixed_points);
            }
            else {
                // out = cdcpd(rgb_image, cur_depth, cur_pc_downsampled, mask, placeholder, template_cloud, false, false, false, 0, fixed_points);
                // std::cout << "pred 0" << std::endl;
                out = cdcpd(cur_image_orig, cur_depth, cur_pc_downsampled, mask, placeholder, template_cloud, one_frame_velocity, one_frame_config, is_grasped, nh_ptr, translation_dir_deformability, translation_dis_deformability, rotation_deformability, true, is_interaction, true, 0, fixed_points);
            }

            
            template_cloud = out.gurobi_output;

            // convert to MatrixXf
            MatrixXf Y = MatrixXf::Zero(num_of_nodes, 3);
            for (int i = 0; i < num_of_nodes; i ++) {
                Y(i, 0) = template_cloud->points[i].x;
                Y(i, 1) = template_cloud->points[i].y;
                Y(i, 2) = template_cloud->points[i].z;
            }

            MatrixXf X = cur_pc_downsampled.getMatrixXfMap().topRows(3).transpose();
            ROS_INFO_STREAM("Number of points in downsampled point cloud: " + std::to_string(X.rows()));

            // calculate node visibility
            // for each node in Y, determine a point in X closest to it
            std::vector<int> visible_nodes = {};
            for (int m = 0; m < Y.rows(); m ++) {
                double shortest_dist = 100000;
                // loop through all points in X
                for (int n = 0; n < X.rows(); n ++) {
                    double dist = (Y.row(m) - X.row(n)).norm();
                    if (dist < shortest_dist) {
                        shortest_dist = dist;
                    }
                }
                if (shortest_dist <= visibility_threshold) {
                    visible_nodes.push_back(m);
                }
            }

            // // print out results
            // std::cout << "=====" << std::endl;
            // for (int i = 0; i < Y.rows(); i ++) {
            //     std::cout << Y(i, 0) << ", " << Y(i, 1) << ", " << Y(i, 2) << "," << std::endl;
            // }
            // std::cout << "=====" << std::endl;

            // projection and pub image
            MatrixXf nodes_h = Y.replicate(1, 1);
            nodes_h.conservativeResize(nodes_h.rows(), nodes_h.cols()+1);
            nodes_h.col(nodes_h.cols()-1) = MatrixXf::Ones(nodes_h.rows(), 1);
            MatrixXf image_coords = (proj_matrix * nodes_h.transpose()).transpose();

            Mat tracking_img;
            tracking_img = 0.5*cur_image_orig + 0.5*cur_image;

            // draw points
            for (int i = 0; i < image_coords.rows(); i ++) {

                int x = static_cast<int>(image_coords(i, 0)/image_coords(i, 2));
                int y = static_cast<int>(image_coords(i, 1)/image_coords(i, 2));

                cv::Scalar point_color;
                cv::Scalar line_color;

                if (std::find(visible_nodes.begin(), visible_nodes.end(), i) != visible_nodes.end()) {
                    point_color = cv::Scalar(0, 150, 255);
                    line_color = cv::Scalar(0, 255, 0);
                }
                else {
                    point_color = cv::Scalar(0, 0, 255);
                    line_color = cv::Scalar(0, 0, 255);
                }

                if (i != image_coords.rows()-1) {
                    cv::line(tracking_img, cv::Point(x, y),
                                        cv::Point(static_cast<int>(image_coords(i+1, 0)/image_coords(i+1, 2)), 
                                                    static_cast<int>(image_coords(i+1, 1)/image_coords(i+1, 2))),
                                        line_color, 5);
                }

                cv::circle(tracking_img, cv::Point(x, y), 7, point_color, -1);
            }

            // add text
            if (updated_opencv_mask && simulated_occlusion) {
                cv::putText(tracking_img, "occlusion", cv::Point(occlusion_corner_j, occlusion_corner_i-10), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 240), 2);
            }

            // publish image
            tracking_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", tracking_img).toImageMsg();

            // publish marker array
            visualization_msgs::MarkerArray results = MatrixXf2MarkerArray(Y, "camera_color_optical_frame", "node_results", {1.0, 150.0/255.0, 0.0, 0.75}, {0.0, 1.0, 0.0, 0.75});
            results_pub.publish(results);

            double time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cur_time_cb).count();
            ROS_WARN_STREAM("Total callback time difference: " + std::to_string(time_diff) + " ms");

            // change frame id
            out.gurobi_output->header.frame_id = frame_id;
            out.original_cloud.header.frame_id = frame_id;
            out.masked_point_cloud.header.frame_id = frame_id;
            out.downsampled_cloud.header.frame_id = frame_id;
            out.cpd_predict->header.frame_id = frame_id;

            auto time = depth_msg->header.stamp;
            pcl_conversions::toPCL(time, out.original_cloud.header.stamp);
            pcl_conversions::toPCL(time, out.masked_point_cloud.header.stamp);
            pcl_conversions::toPCL(time, out.downsampled_cloud.header.stamp);
            pcl_conversions::toPCL(time, out.gurobi_output->header.stamp);
            // pcl_conversions::toPCL(time, template_cloud_init->header.stamp);
            pcl_conversions::toPCL(time, out.cpd_predict->header.stamp);

            // for synchronized evaluation
            pcl::PCLPointCloud2 result_pc_pclpoincloud2;
            sensor_msgs::PointCloud2 result_pc;
            pcl::toPCLPointCloud2(*(out.gurobi_output), result_pc_pclpoincloud2);
            pcl_conversions::moveFromPCL(result_pc_pclpoincloud2, result_pc);
            result_pc.header = depth_msg->header;

            original_publisher.publish(out.original_cloud);
            masked_publisher.publish(out.masked_point_cloud);
            downsampled_publisher.publish(out.downsampled_cloud);
            pred_publisher.publish(out.cpd_predict);
            output_publisher.publish(result_pc);
        }
    }

    return tracking_img_msg;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "cdcpd2_test");
    ros::NodeHandle nh;

    // load parameters
    nh.getParam("/cdcpd2/num_of_nodes", num_of_nodes);
    nh.getParam("/cdcpd2/is_gripper_info", is_gripper_info);
    nh.getParam("/cdcpd2/use_real_gripper", use_real_gripper);
    nh.getParam("/cdcpd2/alpha", alpha);
    nh.getParam("/cdcpd2/beta", beta);
    nh.getParam("/cdcpd2/lambda", lambda);
    nh.getParam("/cdcpd2/zeta", zeta);
    nh.getParam("/cdcpd2/downsample_leaf_size", downsample_leaf_size);
    nh.getParam("/cdcpd2/k_spring", k_spring);
    nh.getParam("/cdcpd2/is_sim", is_sim);
    nh.getParam("/cdcpd2/is_rope", is_rope);
    nh.getParam("/cdcpd2/translation_dir_deformability", translation_dir_deformability);
    nh.getParam("/cdcpd2/translation_dis_deformability", translation_dis_deformability);
    nh.getParam("/cdcpd2/rotation_deformability", rotation_deformability);
    nh.getParam("/cdcpd2/left_x", left_x);
    nh.getParam("/cdcpd2/left_y", left_y);
    nh.getParam("/cdcpd2/left_z", left_z);
    nh.getParam("/cdcpd2/right_x", right_x);
    nh.getParam("/cdcpd2/right_y", right_y);
    nh.getParam("/cdcpd2/right_z", right_z);
    nh.getParam("/cdcpd2/bag_file", bag_file);
    nh.getParam("/cdcpd2/bag_rate", bag_rate);

    nh.getParam("/cdcpd2/multi_color_dlo", multi_color_dlo);

    nh.getParam("/cdcpd2/camera_info_topic", camera_info_topic);
    nh.getParam("/cdcpd2/rgb_topic", rgb_topic);
    nh.getParam("/cdcpd2/depth_topic", depth_topic);

    nh.getParam("/cdcpd2/hsv_threshold_upper_limit", hsv_threshold_upper_limit);
    nh.getParam("/cdcpd2/hsv_threshold_lower_limit", hsv_threshold_lower_limit);
    nh.getParam("/cdcpd2/visibility_threshold", visibility_threshold);

    // update color thresholding upper bound
    std::string rgb_val = "";
    for (int i = 0; i < hsv_threshold_upper_limit.length(); i ++) {
        if (hsv_threshold_upper_limit.substr(i, 1) != " ") {
            rgb_val += hsv_threshold_upper_limit.substr(i, 1);
        }
        else {
            upper.push_back(std::stoi(rgb_val));
            rgb_val = "";
        }
        
        if (i == hsv_threshold_upper_limit.length()-1) {
            upper.push_back(std::stoi(rgb_val));
        }
    }

    // update color thresholding lower bound
    rgb_val = "";
    for (int i = 0; i < hsv_threshold_lower_limit.length(); i ++) {
        if (hsv_threshold_lower_limit.substr(i, 1) != " ") {
            rgb_val += hsv_threshold_lower_limit.substr(i, 1);
        }
        else {
            lower.push_back(std::stoi(rgb_val));
            rgb_val = "";
        }
        
        if (i == hsv_threshold_lower_limit.length()-1) {
            upper.push_back(std::stoi(rgb_val));
        }
    }

    std::cout << "before callback" << std::endl;

    nh_ptr = std::make_shared<ros::NodeHandle>(nh);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber opencv_mask_sub = it.subscribe("/mask_with_occlusion", 10, update_opencv_mask);
    image_transport::Publisher tracking_img_pub = it.advertise("/cdcpd2/results_img", 10);
    results_pub = nh.advertise<visualization_msgs::MarkerArray>("/results_marker", 1);
    camera_info_sub = nh.subscribe(camera_info_topic, 1, update_camera_info);

    original_publisher = nh.advertise<PointCloud> ("cdcpd/original", 1);
    masked_publisher = nh.advertise<PointCloud> ("cdcpd/masked", 1);
    downsampled_publisher = nh.advertise<PointCloud> ("cdcpd/downsampled", 1);
    pred_publisher = nh.advertise<PointCloud>("cdcpd/prediction", 1);
    output_publisher;
    if (is_gripper_info) {
        output_publisher = nh.advertise<PointCloud> ("cdcpd2_results_pc", 1);
    }
    else {
        output_publisher = nh.advertise<PointCloud> ("cdcpd2_no_gripper_results_pc", 1);
    }

    tf2_ros::TransformListener tfListener(tfBuffer);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, rgb_topic, 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, depth_topic, 10);
    // message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub(nh, "/camera/depth/color/points", 10);
    // message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> sync(image_sub, depth_sub, pc_sub, 10);

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
            // sensor_msgs::ImagePtr tracking_img = Callback(img_msg, depth_msg, pc_msg); // sensor_msgs::ImagePtr tracking_img = 
            sensor_msgs::ImagePtr tracking_img = Callback(img_msg, depth_msg);
            tracking_img_pub.publish(tracking_img);
        }
    );
    
    ros::spin();
}