# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import sys

import cv2
import maxflow
import numpy as np
from scipy.spatial import Delaunay
from skimage.segmentation import slic


def help_message():
    print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
    print("[Input_Image]")
    print("Path to the input image")
    print("[Output_Directory]")
    print("Output directory")
    print(sys.argv[0] + " astronaut.png" + " ./")


drawing = False  # true if mouse is pressed
mode = True  # if True, draw background. Press 'b' to toggle to foreground
ix, iy = -1, -1


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global img, ix, iy, drawing, mode, mask, disp_img, overlay, masked_img, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.line(disp_img, (ix, iy), (x, y), (255, 0, 0), 5)
                cv2.line(overlay, (ix, iy), (x, y), (255, 0, 0), 5)
            else:
                cv2.line(disp_img, (ix, iy), (x, y), (0, 0, 255), 5)
                cv2.line(overlay, (ix, iy), (x, y), (0, 0, 255), 5)
            ix, iy = x, y
            mask = calculate_mask(overlay)
            cv2.imshow('output', mask)
            cv2.imshow('image', disp_img)
            frame = np.hstack((disp_img, mask))
            out.write(frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.line(disp_img, (ix, iy), (x, y), (255, 0, 0), 5)
            cv2.line(overlay, (ix, iy), (x, y), (255, 0, 0), 5)
        else:
            cv2.line(disp_img, (ix, iy), (x, y), (0, 0, 255), 5)
            cv2.line(overlay, (ix, iy), (x, y), (0, 0, 255), 5)
        ix, iy = -1, -1


# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=19)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [25, 25]  # H = S = 20
    ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
    colors_hists = np.float32(
        [cv2.calcHist([hsv], [0, 1], np.uint8(segments == i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers, colors_hists, segments, tri.vertex_neighbor_vertices)


# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:, :, 0] != 255])
    bg_segments = np.unique(superpixels[marking[:, :, 2] != 255])
    return (fg_segments, bg_segments)


# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids], axis=0)
    return h / h.sum()


# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask


# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])


# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr) - 1):
        N = indices[indptr[i]:indptr[i + 1]]  # list of neighbor superpixels
        hi = norm_hists[i]  # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]  # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 25 - cv2.compareHist(hi, hn, hist_comp_alg),
                       25 - cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i, h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000)  # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0)  # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                        cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)


def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0] ** (1 / 2.0)

        return total_diff;


def calculate_mask(overlay):
    fg_segments, bg_segments = find_superpixels_under_marking(overlay, superpixels)

    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

    graph_cut = do_graph_cut((fg_cumulative_hist, bg_cumulative_hist), (fg_segments, bg_segments), norm_hists,
                             neighbors)

    mask_img = np.where(graph_cut[superpixels], 255, 0).astype('uint8')
    return cv2.merge((mask_img, mask_img, mask_img))


def init_superpixels():
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)
    norm_hists = normalize_histograms(color_hists)
    return centers, color_hists, superpixels, neighbors, norm_hists


if __name__ == '__main__':
    # validate the input arguments
    if (len(sys.argv) != 3):
        help_message()
        sys.exit()

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    # ======================================== #
    # write all your codes here
    # standard assignment
    centers, color_hists, superpixels, neighbors, norm_hists = init_superpixels()

    disp_img = img.copy()

    # bonus
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    cv2.imshow('image', disp_img)
    overlay = np.ones_like(disp_img, dtype='uint8') * 255

    frame = np.hstack((disp_img, overlay))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_name = sys.argv[2] + "bonus.mp4"
    out = cv2.VideoWriter(output_name, fourcc, 15.0, (frame.shape[1], frame.shape[0]))

    mask = np.ones_like(disp_img, dtype='uint8')
    masked_img = cv2.merge((mask, mask, mask))
    cv2.imshow('output', mask)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            mode = not mode
        elif k == 27:
            break
    out.release()
    cv2.destroyAllWindows()
    # ======================================== #
