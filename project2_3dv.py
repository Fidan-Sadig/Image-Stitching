import cv2
import numpy as np
import os
import json
import zipfile

def load_images(image_paths):
    """Loading images from the given paths"""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image: {path}")
        else:
            print(f"Loaded image: {path} with shape: {img.shape}")
            images.append(img)
    return images

def resize_images(images, width=800):
    """Resizing images to have the same width"""
    resized = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        new_height = int(width / aspect_ratio)
        resized.append(cv2.resize(img, (width, new_height)))
    return resized

def stitch_images(images):
    """Using OpenCV's built-in stitcher to create a panorama"""
    stitcher = cv2.Stitcher_create()  # Using createStitcher() for older OpenCV versions
    status, panorama = stitcher.stitch(images)

    print(f"Stitching status: {status}")

    if status == cv2.Stitcher_OK:
        print("Panorama created successfully!")
        return panorama
    else:
        print("Error during stitching.")
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Error: More images are needed.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Error: Homography estimation failed.")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Error: Camera parameters adjustment failed.")
        return None

def crop_panorama(panorama):
    """Croping the panorama to remove black edges"""
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return panorama[y:y+h, x:x+w]

def detect_and_match_features(images):
    """Detecting and matching features between images using ORB"""
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []
    matches_list = []

    # Detecting keypoints and descriptors
    for img in images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Matching features between consecutive images
    for i in range(len(images) - 1):
        raw_matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        matches_list.append(good_matches)

    return keypoints_list, matches_list

def save_coordinates_and_image(result, keypoints_list, matches_list):
    """Saving coordinates of matched points and final panorama image"""

    # Define the target folder where the files should be saved
    save_folder = "/Users/fidansadigova/Downloads/3dvisionprojects 2"

    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Define full paths for the output files
    output_image_path = os.path.join(save_folder, "panorama1.jpg")
    output_coordinates_path = os.path.join(save_folder, "coordinates.json")
    zip_filename = os.path.join(save_folder, "panorama_with_coordinates.zip")

    # Save the panorama image
    cv2.imwrite(output_image_path, result)

    # Save the coordinates in a JSON file
    coordinates = []
    for i in range(len(matches_list)):
        src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]]).reshape(-1, 2)
        coordinates.append((src_pts.tolist(), dst_pts.tolist()))

    with open(output_coordinates_path, 'w') as f:
        json.dump(coordinates, f, indent=4)

    # Create a ZIP file containing the image and coordinates
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(output_image_path, os.path.basename(output_image_path))
        zipf.write(output_coordinates_path, os.path.basename(output_coordinates_path))

    print(f"✅ Panorama and files saved in: {save_folder}")

    return zip_filename

def main():
    # Hardcoded paths to the 3 images
    image_paths = [
        '/Users/fidansadigova/Downloads/1_3dv/DSCF8658.jpeg',
        '/Users/fidansadigova/Downloads/1_3dv/DSCF8659.jpeg',
        '/Users/fidansadigova/Downloads/1_3dv/DSCF8660.jpeg'
    ]

    # Loading images
    images = load_images(image_paths)
    if len(images) < 2:
        print("Please provide at least two images for stitching.")
        return

    # Resizing images
    images = resize_images(images)

    # Detecting and matching features
    keypoints_list, matches_list = detect_and_match_features(images)

    # Stitching images
    result = stitch_images(images)

    # Croping and saving result
    if result is not None:
        result = crop_panorama(result)
        zip_filename = save_coordinates_and_image(result, keypoints_list, matches_list)
        print(f"Files saved in ZIP archive: {zip_filename}")
    else:
        print("Panorama creation failed. No files saved.")


if __name__ == "__main__":
    main()
