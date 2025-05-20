import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def stitch_images(img1_path, img2_path, output_path):
    """
    Stitch two images together to create a panoramic image that spans the full combined length
    of both images, either horizontally or vertically based on the detected features.
    
    Parameters:
    img1_path (str): Path to the first image
    img2_path (str): Path to the second image
    output_path (str): Path to save the output panoramic image
    """
    print("Step 1: Loading input images...")
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images. Please check file paths.")
        return None
    
    # Store original images for later
    orig_img1 = img1.copy()
    orig_img2 = img2.copy()
    
    # Convert images to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    print("Step 2: Detecting keypoints...")
    # Using SIFT for feature detection
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    print(f"Found {len(keypoints1)} keypoints in image 1")
    print(f"Found {len(keypoints2)} keypoints in image 2")
    
    print("Step 3: Matching features using FLANN...")
    # FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Get k=2 nearest matches
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    print("Step 4: Applying Lowe's Ratio Test to filter matches...")
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches after Lowe's ratio test")
    
    # Minimum matches needed
    MIN_MATCH_COUNT = 10
    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"Not enough good matches: {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None
    
    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Determine if horizontal or vertical alignment
    # Calculate the average displacement in x and y directions
    displacements = dst_pts.reshape(-1, 2) - src_pts.reshape(-1, 2)
    avg_dx = np.mean(np.abs(displacements[:, 0]))
    avg_dy = np.mean(np.abs(displacements[:, 1]))
    
    # Determine if images are aligned horizontally or vertically
    is_horizontal = avg_dx > avg_dy
    orientation = "horizontally" if is_horizontal else "vertically"
    print(f"Images appear to be aligned {orientation}")
    
    print("Step 5: Estimating homography matrix...")
    # Find homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    
    # Count inliers (matches that survived RANSAC)
    inliers = sum(matches_mask)
    print(f"Matches that passed RANSAC (inliers): {inliers}/{len(good_matches)}")
    
    print("Step 6: Warping image using homography...")
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Define the corners of the first image
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Transform corners of img1 into img2's space
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    
    # Calculate the full bounds of the combined images
    all_corners = np.vstack([
        np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2),
        corners1_transformed
    ])
    
    # Find the min and max coordinates to determine the size of our panorama
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Create translation matrix to adjust for negative offsets
    trans_x = -x_min if x_min < 0 else 0
    trans_y = -y_min if y_min < 0 else 0
    translation_dist = [trans_x, trans_y]
    
    # Create the translation matrix
    translation_matrix = np.array([
        [1, 0, trans_x],
        [0, 1, trans_y],
        [0, 0, 1]
    ])
    
    # Apply the translation to the homography matrix
    full_homography = translation_matrix.dot(H)
    
    # Calculate the output image dimensions
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    
    # Ensure positive dimensions
    if panorama_width <= 0 or panorama_height <= 0:
        print("Error: Invalid panorama dimensions. Check homography calculation.")
        return None
    
    print(f"Output panorama dimensions: {panorama_width}x{panorama_height}")
    
    print("Step 7: Blending and stitching images...")
    # Create a canvas for the output panorama
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    
    # Warp first image to create the panorama according to the homography
    warped_img1 = cv2.warpPerspective(img1, full_homography, (panorama_width, panorama_height))
    
    # Place the second image on the canvas with the proper offset
    # Make sure we don't go out of bounds (handle the broadcasting issue)
    y_offset = trans_y
    x_offset = trans_x
    y_end = min(y_offset + h2, panorama_height)
    x_end = min(x_offset + w2, panorama_width)
    
    # Calculate how much of img2 we can fit
    h2_visible = y_end - y_offset
    w2_visible = x_end - x_offset
    
    if h2_visible > 0 and w2_visible > 0:
        panorama[y_offset:y_end, x_offset:x_end] = img2[:h2_visible, :w2_visible]
    
    # Create a mask for the warped image (where we have data)
    warped_gray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    _, warped_mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Create another mask for the placed second image
    img2_mask = np.zeros((panorama_height, panorama_width), dtype=np.uint8)
    if h2_visible > 0 and w2_visible > 0:
        img2_mask[y_offset:y_end, x_offset:x_end] = 255
    
    # Identify the overlap region
    overlap_mask = cv2.bitwise_and(warped_mask, img2_mask)
    
    # If there's an overlap, apply a gradual blend (feathering)
    if np.any(overlap_mask):
        # Create a gradual blend in the overlap region
        # Use distance transform to create a gradient
        dist = cv2.distanceTransform(overlap_mask, cv2.DIST_L2, 3)
        # Fixed line: Get min and max values from distance transform
        min_val, max_dist, min_loc, max_loc = cv2.minMaxLoc(dist)
        
        # Normalize to create an alpha blend weight
        if max_dist > 0:
            blend_weight = dist / max_dist
        else:
            blend_weight = dist
        
        # Expand to 3 channels for color blending
        blend_weight_3d = np.stack([blend_weight] * 3, axis=2)
        
        # Apply blend in the overlap region
        overlap_indices = np.where(overlap_mask > 0)
        for y, x in zip(overlap_indices[0], overlap_indices[1]):
            alpha = blend_weight[y, x]
            panorama[y, x] = (1 - alpha) * panorama[y, x] + alpha * warped_img1[y, x]
        
        # For non-overlapping regions, just take the respective image data
        non_overlap_warped = cv2.bitwise_and(warped_mask, cv2.bitwise_not(img2_mask))
        for c in range(3):
            panorama[:, :, c] = np.where(
                non_overlap_warped > 0,
                warped_img1[:, :, c],
                panorama[:, :, c]
            )
    else:
        # No overlap, just combine the images
        for c in range(3):
            panorama[:, :, c] = np.where(
                warped_mask > 0,
                warped_img1[:, :, c],
                panorama[:, :, c]
            )
    
    print("Saving panoramic image...")
    cv2.imwrite(output_path, panorama)
    print(f"Panoramic image saved to {output_path}")
    
    # For visualization
    rgb_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)
    
    # Visualize results
    plt.figure(figsize=(20, 12))
    
    plt.subplot(221)
    plt.imshow(img1_rgb)
    plt.title('Image 1')
    
    plt.subplot(222)
    plt.imshow(img2_rgb)
    plt.title('Image 2')
    
    # Display inlier matches
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
    match_img = cv2.drawMatches(orig_img1, keypoints1, orig_img2, keypoints2, 
                            inlier_matches[:30], None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(223)
    plt.imshow(match_img_rgb)
    plt.title(f'Feature Matches ({inliers} inliers)')
    
    plt.subplot(224)
    plt.imshow(rgb_panorama)
    plt.title(f'Panorama Result ({orientation} aligned)')
    
    plt.tight_layout()
    plt.savefig('panorama_process.png')
    
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display plot: {e}")
    
    return rgb_panorama


def create_panorama(images_paths, output_path):
    """
    Create a panorama from multiple images by stitching them in sequence.
    
    Parameters:
    images_paths (list): List of paths to images in sequence
    output_path (str): Path to save the final panoramic image
    """
    if len(images_paths) < 2:
        print("Error: At least two images are required for stitching.")
        return None
    
    # Start with the first image as our base
    result_path = images_paths[0]
    
    # Process images in sequence
    for i in range(1, len(images_paths)):
        # Create a temporary output path
        temp_output = f"temp_panorama_{i}.jpg"
        
        # Stitch the current result with the next image
        print(f"\nStitching image {i+1}/{len(images_paths)}...")
        result = stitch_images(result_path, images_paths[i], temp_output)
        
        if result is None:
            print(f"Failed to stitch image {images_paths[i]}. Stopping.")
            return None
        
        # Update the result path
        result_path = temp_output
    
    # Rename the final result
    os.rename(result_path, output_path)
    print(f"\nFinal panorama saved to: {output_path}")
    
    # Clean up temporary files
    for i in range(1, len(images_paths)):
        temp_file = f"temp_panorama_{i}.jpg"
        if os.path.exists(temp_file) and temp_file != output_path:
            os.remove(temp_file)
    
    return cv2.imread(output_path)


if __name__ == "__main__":
    
    output_path = "panorama_result.jpg"
    images=input("Enter Path of All Images: ").split()

    # For multiple images in sequence
    if (len(images) > 2) :
        result = create_panorama(images, output_path)

    # For two images
    else:
        result = stitch_images(images[1], images[2], output_path)
            
    if result is not None:
        print("Panorama created successfully!")
    else:
        print("Failed to create panorama.")