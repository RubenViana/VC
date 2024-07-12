import cv2
import numpy as np
import json
import sys

def input(inputs):
    with open(inputs, 'r') as file:
        data = json.load(file)

    list = []
    for id, img in enumerate(data["image_files"]):
        list.append({"file_name": img, "num_colors": 0, "num_detections": 0, "detected_objects": []})

    return list

def output(results, output_file):
    with open(output_file, 'w') as file:
        json.dump({"results": results}, file, indent=4)

def intersection_area_cnt(cnt1, cnt2, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blank1 = np.zeros_like(gray)
    blank2 = np.zeros_like(gray)

    cv2.drawContours(blank1, [cnt1], 0, (255), thickness=cv2.FILLED)
    cv2.drawContours(blank2, [cnt2], 0, (255), thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(blank1, blank2)

    contours, _ = cv2.findContours(
        intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intersection_area = 0
    for contour in contours:
        intersection_area += cv2.contourArea(contour)
    return intersection_area

def calculate_area(box):
    return box[2] * box[3]

def intersection_area(box1, box2):
    x_overlap = max(0, min(box1[0]+box1[2], box2[0]+box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[1]+box1[3], box2[1]+box2[3]) - max(box1[1], box2[1]))
    return x_overlap * y_overlap

def is_corner_of_image(corner, image_size=(600, 600), threshold=20):
    x, y = corner
    image_width, image_height = image_size

    # Check if the corner is within the threshold range of any of the corners of the image
    if (0 <= x <= threshold and 0 <= y <= threshold) or \
       (0 <= x <= threshold and image_height - threshold <= y <= image_height) or \
       (image_width - threshold <= x <= image_width and 0 <= y <= threshold) or \
       (image_width - threshold <= x <= image_width and image_height - threshold <= y <= image_height):
        return True
    else:
        return False

def remove_overlapping_boxes(cnts):
    cnts = sorted(cnts, key=lambda cnt: calculate_area(cv2.boundingRect(cnt)), reverse=True)  # Sort cnts by area in descending order
    overlapping_indices = set()

    for i in range(len(cnts)):
        if i in overlapping_indices:
            continue

        for j in range(i + 1, len(cnts)):
            if j in overlapping_indices:
                continue

            if intersection_area(cv2.boundingRect(cnts[i]), cv2.boundingRect(cnts[j])) >= 0.9 * min(calculate_area(cv2.boundingRect(cnts[i])), calculate_area(cv2.boundingRect(cnts[j]))):
                overlapping_indices.add(j)

    # remove table corners of the image
    for i, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        if is_corner_of_image(top_left) or is_corner_of_image(top_right) or is_corner_of_image(bottom_left) or is_corner_of_image(bottom_right):
            overlapping_indices.add(i)

    return [cnts[i] for i in range(len(cnts)) if i not in overlapping_indices]

def most_used_color(image, range_radius=5):
    # Ensure image is in the correct format (RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image to a 1D array of pixels
    pixels = rgb_image.reshape((-1, 3))

    # Calculate histogram of colors
    hist = cv2.calcHist([rgb_image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # Exclude black color (0, 0, 0) from consideration
    hist[0, 0, 0] = 0

    # Find the index of the bin with the highest frequency
    most_used_index = np.unravel_index(np.argmax(hist), hist.shape)

    # Convert bin index to BGR color
    most_used_color = np.array(most_used_index[::-1], dtype=np.uint8)

    return most_used_color

def display_image_info(img, image, bbs):
    # create a window to display the image
    cv2.namedWindow(img["file_name"], cv2.WINDOW_AUTOSIZE)

    # Draw bounding boxes
    for bb in bbs:
        cv2.rectangle(image, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0, 255, 0), 2)

    # Display the number of colors and detections
    cv2.putText(image, "num_colors: " + str(img["num_colors"]), (5, 564), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "num_detections: " + str(img["num_detections"]), (5, 592), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow(img["file_name"], image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def count_colors(image, cnts):
    colors = []
    for cnt in cnts:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, (255), -1)
        img = cv2.bitwise_and(image, image, mask=mask)
        color = most_used_color(img)

        # if color or simmilar color is already in the list, skip
        if len(colors) == 0:
            colors.append(color)
        else:
            for c in colors:
                diff = np.abs(np.array(c, dtype=np.int16) - np.array(color, dtype=np.int16))
                if np.all(diff < 25):
                    break
            else:
                colors.append(color)

    return len(colors)

def count_bricks(image):
    blur = cv2.medianBlur(image, 11)

    # Canny
    canny = cv2.Canny(blur, 40, 210)
    canny = cv2.dilate(canny, None, iterations=1)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Remove overlapping bounding boxes in contours
    contours = remove_overlapping_boxes(contours)

    # Bounding boxes
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    return contours, bounding_boxes

def process_image(img):
    # Read image
    image = cv2.imread(img["file_name"])

    # Image size
    scale_x = image.shape[1] / 600
    scale_y = image.shape[0] / 600

    # Resize image to fit the screen
    image = cv2.resize(image, (600, 600))

    # Number of bricks
    cnts, bbs = count_bricks(image)
    img["num_detections"] = len(cnts)

    # Bounding boxes
    for bb in bbs:
        img["detected_objects"].append({"xmin": int(bb[0] * scale_x), "ymin": int(bb[1] * scale_y), "xmax": int((bb[0]+bb[2] * scale_x)), "ymax": int((bb[1]+bb[3]) * scale_y)})

    # Number of colors
    img["num_colors"] = count_colors(image, cnts)

    # Display image info
    display_image_info(img, image, bbs)
    

def image_processing_pipeline(images):
    for img in images:
        process_image(img)

    return images


def main(input_file, output_file):
    # Load images
    images = input(input_file)

    # Run pipeline
    results = image_processing_pipeline(images)

    # Output results
    output(results, output_file)



if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)
    else:
        print("Usage: python ipp.py <input_file> <output_file>")
        sys.exit(1)


