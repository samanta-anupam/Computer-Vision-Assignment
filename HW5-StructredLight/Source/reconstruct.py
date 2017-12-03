# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import pickle
import sys

import cv2
import numpy as np

camera_points = []
projector_points = []
masked_rgb_points = []


def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    global camera_points, projector_points, masked_rgb_points
    # load the prepared stereo calibration between projector and camera

    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_img = cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR)
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2)), (0, 0), fx=scale_factor, fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                               fx=scale_factor,
                               fy=scale_factor)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        scan_bits = scan_bits + on_mask * bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)
    camera_points = []
    projector_points = []
    camera_points_rgb = []
    img_output = np.zeros((960, 1280, 3))
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code
            x_p, y_p = binary_codes_ids_codebook[scan_bits[y, x]]
            if x_p >= 1279 or y_p >= 799:  # filter
                continue
            projector_points.append((x_p, y_p))
            camera_points_rgb.append(ref_img[y, x])
            img_output[y, x, 2] = np.uint8(x_p * 255.0 / 1279.0)
            img_output[y, x, 1] = np.uint8(y_p * 255.0 / 799.0)
            camera_points.append((x / 2.0, y / 2.0))
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    cv2.imwrite(sys.argv[1] + 'correspondence.jpg', img_output)

    camera_points_rgb = np.array(camera_points_rgb)

    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

        camera_pts = np.array(camera_points, dtype=np.float32)
        num_pts = camera_pts.size / 2
        camera_pts.shape = (num_pts, 1, 2)
        # print(camera_pts.shape)
        camera_points = cv2.undistortPoints(src=camera_pts, cameraMatrix=camera_K, distCoeffs=camera_d)

        project_pts = np.array(projector_points, dtype=np.float32)
        num_pts = project_pts.size / 2
        project_pts.shape = (num_pts, 1, 2)
        # print(project_pts.shape)
        projector_points = cv2.undistortPoints(src=project_pts, cameraMatrix=projector_K, distCoeffs=projector_d)

        projMatr1 = np.hstack((np.identity(3), np.zeros((3, 1))))
        projMatr2 = np.hstack((projector_R, projector_t))
        homogeneous_3d_points = cv2.triangulatePoints(projMatr1=projMatr1,
                                                      projMatr2=projMatr2, projPoints1=camera_points,
                                                      projPoints2=projector_points).T

        points_3d = cv2.convertPointsFromHomogeneous(homogeneous_3d_points)

        mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400)
        pts_3d = []
        c_pts = []
        for i, b in enumerate(mask):
            if b:
                c_pts.append(camera_points_rgb[i])
                pts_3d.append(points_3d[i])

        masked_rgb_points = np.array(c_pts)
        points_3d = np.array(pts_3d)
        write_3d_points_color(points_3d)
        return points_3d


def write_3d_points_color(points_3d):
    output_name = sys.argv[1] + "output_color.xyzrgb"
    with open(output_name, "w") as f:
        for i, p in enumerate(points_3d):
            r, g, b = masked_rgb_points[i]
            f.write("%d %d %d %d %d %d\n" % (p[0][0], p[0][1], p[0][2], b, g, r))


def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")

    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

    return points_3d, camera_points, projector_points


if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
