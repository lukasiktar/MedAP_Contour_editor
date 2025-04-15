import cv2
import numpy as np

from Segmentation_helper import create_directory

#Polygon segmentator class
class Polygon_Segmentator:
    def __init__(self, image, file_name, real_image_shape,polygon_points, mask) -> None:
        """
        Polygon segmentation instance.

        Args:
            image - an image to be segmented (zoomed)
            file_name - the image name stored from the image file
            real_image_shape - real_image_shape [width, height]
            number_of_polygons - the specified number of polygons to be created
            polygon_points - polygon points in a list

        Output:
            No output. 
        """

        self.image=image
        self.file_name=file_name
        self.real_image_shape=real_image_shape
        self.polygon_points=polygon_points
        self.mask=mask

    def setup_directories(self) -> None:
        '''
        Creates directories needed for setup, masks, annotations
        and txt files.
        '''
        create_directory('AnnotatedDataset')
        create_directory('AnnotatedDataset/masks')
        create_directory('AnnotatedDataset/annotations')
        create_directory('AnnotatedDataset/txt')
        
    def create_polygon(self) -> None:
        '''
        Creates a single polygon and polygon mask
        '''

        # Ensure polygon vertices are correct
        print("Polygon vertices:", self.polygon_points)
        
        # Convert polygon points to NumPy array format for OpenCV
        self.polygon_vertices = np.array(self.polygon_points, dtype=np.int32).reshape((-1, 1, 2))

        # Fill the polygon on the mask with white (255)
        cv2.fillPoly(self.mask, [self.polygon_vertices], 255)

        # Check if the mask contains any non-zero values (i.e., polygon is drawn)
        if not np.any(self.mask):
            print("Warning: The polygon might be outside the mask boundaries or not visible.")
        else:
            print("Polygon has been successfully drawn on the mask.")

        # Convert mask to boolean format
        self.mask = self.mask > 0

        # Resize the boolean mask to original image size and convert it to uint8 for saving
        self.resized_mask = cv2.resize(self.mask.astype(np.uint8), 
                                    (self.real_image_shape[0], self.real_image_shape[1]),  # (width, height)
                                    interpolation=cv2.INTER_NEAREST)
        print(f"Mask shape: {self.mask.shape}")
        print(f"Resized image mask shape: {self.resized_mask.shape}")
        # Save the mask
        #mask_save_path = f"AnnotatedDataset/masks/{self.file_name}_mask.png"
        #cv2.imwrite(mask_save_path, (self.resized_mask * 255).astype(np.uint8))


