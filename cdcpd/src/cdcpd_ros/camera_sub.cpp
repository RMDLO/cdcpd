#include "cdcpd_ros/camera_sub.h"

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/image_encodings.h>

namespace sm = sensor_msgs;

CameraSubSetup::CameraSubSetup(const std::string& rgb_topic, const std::string& depth_topic,const std::string& info_topic)
    : nh(),
      pnh("~"),
      hints("raw", ros::TransportHints(), pnh),
      queue_size(10),
      rgb_topic(rgb_topic),
      depth_topic(depth_topic),
      info_topic(info_topic),
      spinner(1, &queue) {
  nh.setCallbackQueue(&queue);
  pnh.setCallbackQueue(&queue);
  spinner.start();
}

KinectSub::KinectSub(const std::function<void(cv::Mat, cv::Mat, cv::Matx33d)>& _externCallback,
                     CameraSubSetup& _options)
    : options(_options),
      externCallback(_externCallback),
      it(options.nh),
      rgb_sub(it, options.rgb_topic, options.queue_size, options.hints),
      depth_sub(it, options.depth_topic, options.queue_size, options.hints),
      cam_sub(options.nh, options.info_topic, options.queue_size),
      sync(SyncPolicy(options.queue_size), rgb_sub, depth_sub, cam_sub) {
  if (_options.hints.getTransport() == "compressed") {
    // TODO: when creating these subscribers, both the rgb and depth try to create a
    // `cdcpd_node/compressed/set_parameters` service, this is presumably not an issue for now, but it is messy
    ROS_INFO("Ignore the 'Tried to advertise a service that is already advertised' ... message, see cpp file.");
  }

  sync.registerCallback(boost::bind(&KinectSub::imageCb, this, _1, _2, _3));
}

void KinectSub::imageCb(const sm::ImageConstPtr& rgb_msg, const sm::ImageConstPtr& depth_msg,
                        const sm::CameraInfoConstPtr& cam_msg) {
  cv_bridge::CvImagePtr cv_rgb_ptr;
  try {
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sm::image_encodings::RGB8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("RGB cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImagePtr cv_depth_ptr;
  try {
    cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Depth cv_bridge exception: %s", e.what());
    return;
  }

  if (depth_msg->encoding != sm::image_encodings::TYPE_16UC1) {
    ROS_INFO_THROTTLE(10, "Depth message is not in %s format. Converting.", sm::image_encodings::TYPE_16UC1.c_str());
    if (depth_msg->encoding == sm::image_encodings::TYPE_32FC1) {
      cv::Mat convertedDepthImg(cv_depth_ptr->image.size(), CV_16UC1);

      const int V = cv_depth_ptr->image.size().height;
      const int U = cv_depth_ptr->image.size().width;

      for (int v = 0; v < V; ++v) {
        for (int u = 0; u < U; ++u) {
          convertedDepthImg.at<uint16_t>(v, u) =
              depth_image_proc::DepthTraits<uint16_t>::fromMeters(cv_depth_ptr->image.at<float>(v, u));
        }
      }

      cv_depth_ptr->encoding = sm::image_encodings::TYPE_16UC1;
      cv_depth_ptr->image = convertedDepthImg;
    } else {
      ROS_ERROR_THROTTLE(10, "Unhandled depth message format %s", depth_msg->encoding.c_str());
      return;
    }
  }

  if (externCallback) {
    image_geometry::PinholeCameraModel cameraModel;
    cameraModel.fromCameraInfo(cam_msg);
    externCallback(cv_rgb_ptr->image, cv_depth_ptr->image, cameraModel.fullIntrinsicMatrix());
  }
}
