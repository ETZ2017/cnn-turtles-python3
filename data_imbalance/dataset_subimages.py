import pandas as pd
from PIL import Image

def decompose_image(image_path, turtles, subimage_size=100, overlap=50):
    """
    Decompose an image into smaller overlapped images and label based on turtle presence.

    :param image_path: Path to the image.
    :param turtles: List of tuples (x, y, width, height) for each turtle in the image.
    :param subimage_size: Size of the subimages (100x100 by default).
    :param overlap: Overlap size between subimages.
    :return: List of tuples containing sub-images and labels.
    """
    image = Image.open(image_path)
    width, height = image.size

    subimages = []
    for y in range(0, height - subimage_size + 1, subimage_size - overlap):
        for x in range(0, width - subimage_size + 1, subimage_size - overlap):
            # Crop the sub-image
            subimage = image.crop((x, y, x + subimage_size, y + subimage_size))

            # Check if the sub-image contains a turtle
            contains_turtle = any(
                x < turtle_x + turtle_width and
                x + subimage_size > turtle_x and
                y < turtle_y + turtle_height and
                y + subimage_size > turtle_y
                for turtle_x, turtle_y, turtle_width, turtle_height in turtles
            )

            label = 1 if contains_turtle else 0
            subimages.append((subimage, label))

    return subimages

# Read CSV file
csv_file = '/home/alvaro.berobide/AI_project/cnn-turtles-python3/data_imbalance/turtle_image_metadata_clean.csv'
df = pd.read_csv(csv_file)

# Loop over each image in the CSV
for _, row in df.iterrows():
    image_path = '/home/alvaro.berobide/AI_project/cnn-turtles-python3/duke_turtles/20150805cr3southernoffshoreregion_20150805_234156_to_20150805_234555/20150805cr3southernoffshoreregion_20150805_234156_IMG_8966_NIR.jpg'
    turtle_coords = [(row['x'], row['y'], 65, 65)]  # Modify as per your CSV structure

    subimages = decompose_image(image_path, turtle_coords)

    # Save subimages or process further
    for i, (subimage, label) in enumerate(subimages):
        subimage.save(f"/home/alvaro.berobide/AI_project/subimages/{image_path}_subimage_{i}_label_{label}.jpg")
