import rospy
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
import ros_numpy
from cdcpd.cdcpd import CDCPDParams, ConstrainedDeformableCPD
from cdcpd.cpd import CPDParams
from cdcpd.optimizer import DistanceConstrainedOptimizer
from cdcpd.geometry_utils import build_line
from cdcpd.cv_utils import chroma_key_rope
from cdcpd.prior import ThresholdVisibilityPrior

import cv2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation as R

def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

# original post: https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def ndarray2MarkerArray (Y, marker_frame, node_color, line_color):
    results = MarkerArray()
    for i in range (0, len(Y)):
        cur_node_result = Marker()
        cur_node_result.header.frame_id = marker_frame
        cur_node_result.type = Marker.SPHERE
        cur_node_result.action = Marker.ADD
        cur_node_result.ns = "node_results" + str(i)
        cur_node_result.id = i

        cur_node_result.pose.position.x = Y[i, 0]
        cur_node_result.pose.position.y = Y[i, 1]
        cur_node_result.pose.position.z = Y[i, 2]
        cur_node_result.pose.orientation.w = 1.0
        cur_node_result.pose.orientation.x = 0.0
        cur_node_result.pose.orientation.y = 0.0
        cur_node_result.pose.orientation.z = 0.0

        cur_node_result.scale.x = 0.01
        cur_node_result.scale.y = 0.01
        cur_node_result.scale.z = 0.01
        cur_node_result.color.r = node_color[0]
        cur_node_result.color.g = node_color[1]
        cur_node_result.color.b = node_color[2]
        cur_node_result.color.a = node_color[3]

        results.markers.append(cur_node_result)

        if i == len(Y)-1:
            break

        cur_line_result = Marker()
        cur_line_result.header.frame_id = marker_frame
        cur_line_result.type = Marker.CYLINDER
        cur_line_result.action = Marker.ADD
        cur_line_result.ns = "line_results" + str(i)
        cur_line_result.id = i

        cur_line_result.pose.position.x = ((Y[i] + Y[i+1])/2)[0]
        cur_line_result.pose.position.y = ((Y[i] + Y[i+1])/2)[1]
        cur_line_result.pose.position.z = ((Y[i] + Y[i+1])/2)[2]

        rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), (Y[i+1]-Y[i])/pt2pt_dis(Y[i+1], Y[i])) 
        r = R.from_matrix(rot_matrix)
        x = r.as_quat()[0]
        y = r.as_quat()[1]
        z = r.as_quat()[2]
        w = r.as_quat()[3]

        cur_line_result.pose.orientation.w = w
        cur_line_result.pose.orientation.x = x
        cur_line_result.pose.orientation.y = y
        cur_line_result.pose.orientation.z = z
        cur_line_result.scale.x = 0.005
        cur_line_result.scale.y = 0.005
        cur_line_result.scale.z = pt2pt_dis(Y[i], Y[i+1])
        cur_line_result.color.r = line_color[0]
        cur_line_result.color.g = line_color[1]
        cur_line_result.color.b = line_color[2]
        cur_line_result.color.a = line_color[3]

        results.markers.append(cur_line_result)
    
    return results

# for creating occlusion manually
occlusion_mask_rgb = None
def update_occlusion_mask(data):
	global occlusion_mask_rgb
	occlusion_mask_rgb = ros_numpy.numpify(data)


# initialize tracker
# kinect_intrinsics = np.array(
#     [1068.842896477257, 0.0, 950.2974736758024, 0.0, 1066.0150152835104, 537.097974092338, 0.0, 0.0, 1.0],
#     dtype=np.float32).reshape((3, 3))
# kinect_intrinsics[:2] /= 2.0
proj_matrix = np.array([[918.359130859375,              0.0, 645.8908081054688], \
                        [             0.0, 916.265869140625,   354.02392578125], \
                        [             0.0,              0.0,               1.0]])

template_verts, template_edges = build_line(0.46, 37)
key_func = chroma_key_rope

prior = ThresholdVisibilityPrior(proj_matrix)
optimizer = DistanceConstrainedOptimizer(template=template_verts, edges=template_edges)

cpd_params = CPDParams(beta=4.0)
cdcpd_params = CDCPDParams(prior=prior, optimizer=optimizer, down_sample_size=150)
cdcpd = ConstrainedDeformableCPD(template=template_verts,
                                 cdcpd_params=cdcpd_params)

# initialize ROS publisher
pub = rospy.Publisher("/cdcpd_tracker/points", PointCloud2, queue_size=10)
results_pub = rospy.Publisher("/results", MarkerArray, queue_size=10)

def callback(msg: PointCloud2):
    global occlusion_mask_rgb

    # converting ROS message to dense numpy array
    data = ros_numpy.numpify(msg)
    arr = ros_numpy.point_cloud2.split_rgb_field(data)
    point_cloud_img = structured_to_unstructured(arr[['x', 'y', 'z']])
    color_img = structured_to_unstructured(arr[['r', 'g', 'b']])
    mask_img = key_func(point_cloud_img, color_img)

    # process opencv mask
    if occlusion_mask_rgb is None:
        occlusion_mask_rgb = np.ones(color_img.shape).astype('uint8')*255
    occlusion_mask = cv2.cvtColor(occlusion_mask_rgb.copy(), cv2.COLOR_RGB2GRAY)
    mask_img = cv2.bitwise_and(mask_img.astype(np.uint8), occlusion_mask.copy())
    mask_img = mask_img.astype(np.bool_)

    # mask_img is 720x1080x1
    # print(type(mask_img[0, 0]))

    # invoke tracker
    tracking_result = cdcpd.step(point_cloud=point_cloud_img,
                                 mask=mask_img,
                                 cpd_param=cpd_params)

    # converting tracking result to ROS message
    if tracking_result.dtype is not np.float32:
        tracking_result = tracking_result.astype(np.float32)
    out_struct_arr = unstructured_to_structured(tracking_result, names=['x', 'y', 'z'])
    pub_msg = ros_numpy.msgify(PointCloud2, out_struct_arr)
    pub_msg.header = msg.header
    pub.publish(pub_msg)

    results = ndarray2MarkerArray(tracking_result, "camera_color_optical_frame", [255, 150, 0, 0.75], [0, 255, 0, 0.75])
    results_pub.publish(results)


def main():
    rospy.init_node('cdcpd_tracker_node')
    # rospy.Subscriber("/kinect2_victor_head/qhd/points", PointCloud2, callback, queue_size=2)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback, queue_size=2)
    rospy.Subscriber('/mask_with_occlusion', Image, update_occlusion_mask)
    rospy.spin()


main()

