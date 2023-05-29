import os
import cv2
import glob
import csv

test_folder = './Test/'
csv_file = './output_vis-osnet_1_ain-0.7983.csv'
output_folder = './test_processed'

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    bounding_boxes = list(reader)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through the images in the test folder
for image_path in glob.glob(test_folder+'*.png'):

    # Load the image
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path).split('.')[0]

    # Find bounding boxes for the current image
    image_boxes = [box for box in bounding_boxes if box['Image_ID'] == filename]

    # Draw bounding boxes on the image
    for box in image_boxes:
        ymin = float(box['ymin'])
        xmin = float(box['xmin'])
        ymax = float(box['ymax'])
        xmax = float(box['xmax'])
        similarity = float(box['similarity'])
        class_label = box['class']

        # Calculate the coordinates of the bounding box
        height, width, _ = image.shape
        xmin_abs = int(xmin * width)
        xmax_abs = int(xmax * width)
        ymin_abs = int(ymin * height)
        ymax_abs = int(ymax * height)

        # Draw the bounding box rectangle on the image
        cv2.rectangle(image, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), (0, 255, 0), 2)

        # Add class label and similarity score above the bounding box
        label = f'{class_label}: {similarity:.2f}'
        cv2.putText(image, label, (xmin_abs, ymin_abs - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image
    output_path = os.path.join(output_folder, f'annotated_{filename}.png')
    cv2.imwrite(output_path, image)