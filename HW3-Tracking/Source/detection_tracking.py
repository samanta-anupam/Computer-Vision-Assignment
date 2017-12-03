import sys

import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def help_message():
    print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
    print("[Question Number]")
    print("1 Camshift")
    print("2 Particle Filter")
    print("3 Kalman Filter")
    print("4 Optical Flow")
    print("[Input_Video]")
    print("Path to the input video")
    print("[Output_Directory]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]


def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


def camshift_tracking(video, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter(file_name + '.mp4', fourcc, 20.0, (int(video.get(3)), int(video.get(4))))

    frameCounter = 0
    # read first frame
    ret, frame = video.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, (c + w / 2), (r + h / 2)))  # Write as 0,pt_x,pt_y

    img2 = cv2.circle(frame, (int(c + w / 2), int(r + h / 2)), 2, (0, 0, 255), -1)
    # output_video.write(img2)
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)

    roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = video.read()
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        rect, track_window = cv2.CamShift(dst, track_window, term_crit)

        pts = cv2.boxPoints(rect)

        pts = np.int0(pts)
        x = (pts[0][0] + pts[2][0]) / 2
        y = (pts[0][1] + pts[2][1]) / 2

        # img2 = cv2.polylines(frame, [pts], True, 255, 1)
        # img2 = cv2.circle(frame, (int(x),int(y)), 2, (0, 0, 255), -1)
        # output_video.write(img2)
        output.write("%d,%d,%d\n" % (frameCounter, x, y))  # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
    # output_video.release()
    cv2.destroyAllWindows()
    output.close()


'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''


def particle_filter_tracking(video, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter(file_name + '.mp4', fourcc, 20.0,
    #                                (int(video.get(3)), int(video.get(4))))

    frameCounter = 0
    # read first frame
    ret, frame = video.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, (c + w / 2), (r + h / 2)))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)

    roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))
    num_particles = 300
    particles = np.ones((num_particles, 2), int) * np.array((c + w / 2, r + h / 2))

    while (True):
        ret, frame = video.read()
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        steps = 20
        np.add(particles, np.random.uniform(-steps, steps, particles.shape), out=particles, casting="unsafe")

        particles = particles.clip(np.zeros(2), np.array((frame.shape[1] - 1, frame.shape[0] - 1))).astype(int)
        eval_particles = dst[particles.T[1], particles.T[0]]
        wts = np.float32(eval_particles.clip(1))
        wts /= np.sum(wts)

        track_pos = np.sum(particles.T * wts, axis=1).astype(int)

        x = track_pos[0]
        y = track_pos[1]

        # for pts in particles:
        frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # output_video.write(frame)
        output.write("%d,%d,%d\n" % (frameCounter, x, y))  # Write as frame_index,pt_x,pt_y

        if 1. / np.sum(wts ** 2) < num_particles / 2.:
            particles = particles[resample(wts), :]

        frameCounter = frameCounter + 1
    # output_video.release()
    cv2.destroyAllWindows()
    output.close()


'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''


def kalman_filter_tracking(video, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter(file_name + '.mp4', fourcc, 20.0,
    #                                (int(video.get(3)), int(video.get(4))))

    frameCounter = 0
    # read first frame
    ret, frame = video.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, (c + w / 2), (r + h / 2)))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)

    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman = cv2.KalmanFilter(4, 2, 0)
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    # use prediction or posterior as your tracking result
    while (True):
        ret, frame = video.read()
        if ret == False:
            break

        c, r, w, h = detect_one_face(frame)
        measurement_valid = not (c, r, w, h) == (0, 0, 0, 0)

        prediction = kalman.predict()

        x = prediction[0][0]
        y = prediction[1][0]

        # obtain measurement

        if measurement_valid:  # e.g. face found
            measurement = np.array([c + w / 2, r + h / 2]).astype(np.float64)
            posterior = kalman.correct(measurement)
            x = posterior[0]
            y = posterior[1]

        # for pts in particles:
        # frame = cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        # output_video.write(frame)
        output.write("%d,%d,%d\n" % (frameCounter, x, y))  # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
    # output_video.release()
    cv2.destroyAllWindows()
    output.close()


def optical_flow_tracker(v, file_name):
    detect_interval = 5
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(file_name + '.mp4', fourcc, 20.0,
                                   (int(video.get(3)), int(video.get(4))))

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.3,
                          minDistance=4,
                          blockSize=3)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(20, 20),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)
    x = int(c + w / 2)
    y = int(r + h / 2)
    mask[y - 20: y + 20, x - 20:x + 20] = 255

    prev_pts = cv2.goodFeaturesToTrack(prev_frame, mask=mask, **feature_params)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, (c + w / 2), (r + h / 2)))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    while (1):
        ret, frame = v.read()
        if ret == False:
            break

        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **lk_params)
        # Select good points
        good_new = next_pts[st == 1]
        # draw the tracks
        x, y = 0, 0
        for i, new in enumerate(good_new):
            a, b = new.ravel()
            x += a
            y += b
            # frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        x /= len(good_new)
        y /= len(good_new)
        frame = cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.imshow('frame', frame)
        prev_frame = next_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        output_video.write(frame)

        output.write("%d,%d,%d\n" % (frameCounter, int(x), int(y)))  # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        # Now update the previous frame and previous points

    output.close()


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracking(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_filter_tracking(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_filter_tracking(video, "output_kalman.txt")
    elif (question_number == 4):
        optical_flow_tracker(video, "output_of.txt")
