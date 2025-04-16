import os
import sys
import cv2
import torch
# import math
import argparse
import numpy as np
from PIL import Image, ImageTk
from constants import *
from scipy.interpolate import splprep, splev
import customtkinter
from natsort import natsorted
from tkinter import Tk, Label, Canvas, filedialog, messagebox, simpledialog
#from tkinter import ttk, Toplevel
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from MUCSNet_Segment import MUCSNet_Segmentator
from Polygon_segmentator import Polygon_Segmentator

class ContourEditor:
    def __init__(self, root: customtkinter.CTk):

        #Segmentation model load
        MODEL_PATH ="MUCSNet.pth"
        self.net = self.load_seg_model(MODEL_PATH)
        #Root setup
        self.root=root
        self.root.title("MedAP Contour Editor")
        self.root.configure(bg=COLOUR_ROOT_BG)
        customtkinter.set_appearance_mode("dark")

        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        #Initialize variables
        self.operational_image=None #Operational image
        self.original_image=None    #Original image
        self.tk_image=None          #Image format for canvas

        self.segmentation_performed=False   #Segmentaiton flag
        self.points_for_segmentation=50

        #Zoom factors
        self.zoom_value = ZOOM_VALUE
        self.zoom_factor = ZOOM_FACTOR
        self.min_zoom = ZOOM_MIN
        self.max_zoom = ZOOM_MAX

        #Original image dimensions
        self.image_shape=None

        #Tkinter font size
        self.font_size=FONT_SIZE

        # Polygon variables
        self.drawing_polygon = False
        self.ready_for_first_polygon = True
        self.polygon_points = []

        #Apperance mode
        customtkinter.set_appearance_mode('dark')

        #Create GUI elements
        self.canvas=Canvas(root, width=GUI_WIDTH, height=GUI_HEIGHT,  bg=COLOUR_CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side="left", padx=40, pady=20)  # Position the canvas on the left side

        # Create a frame for the buttons on the right side
        button_frame =customtkinter.CTkFrame(root)
        button_frame.pack(side="right", fill="y", padx=30)

        #Buttons
        self.load_button = customtkinter.CTkButton(button_frame,text="Load Dataset", font=(self.font_size,self.font_size), command=self.load_images)          
        self.save_button = customtkinter.CTkButton(button_frame, text="Save Annotation", font=(self.font_size,self.font_size), fg_color='green', hover_color="dark green", command=self.save_image)
        self.reset_button = customtkinter.CTkButton(button_frame, text="Reset Annotation", font=(self.font_size,self.font_size), command=self.reset_rectangle)
        self.draw_polygon_button = customtkinter.CTkButton(button_frame, text="Draw Polygon", font=(self.font_size,self.font_size), command=self.start_polygon_drawing)
        self.perform_segmentation_button = customtkinter.CTkButton(button_frame, text="Perform segmentation", font=(self.font_size,self.font_size), command=self.perform_segmentation)
        self.draw_empty_segmetation_button=customtkinter.CTkButton(button_frame, text="Empty Segmentation", font=(self.font_size,self.font_size), command=self.perform_empty_mask_segmentation)
        self.exit_button = customtkinter.CTkButton(button_frame, text="Exit MedAP", font=(self.font_size,self.font_size), fg_color='red', hover_color="dark red", command=root.quit)

        self.undo_button = customtkinter.CTkButton(button_frame, text="Fix previous", font=(self.font_size,self.font_size), fg_color='medium slate blue', hover_color="dark slate blue", command=self.del_prev_image)

        # Arrange these buttons in the grid (1 column, multiple rows)
        self.load_button.grid(row=0, column=0, ipadx=12, ipady=12, padx=20, pady=10,sticky="ew")
        self.save_button.grid(row=1, column=0, ipadx=12, ipady=12, padx=20, pady=20,sticky="ew")
        self.reset_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.draw_polygon_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.perform_segmentation_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.draw_empty_segmetation_button.grid(row=5, column=0, padx=20, pady=20, sticky="ew")
        self.exit_button.grid(row=6, column=0, ipadx=12, ipady=12, padx=20, pady=30, sticky="ew")
        self.undo_button.grid(row=8, column=0, ipadx=0, ipady=12, padx=20, pady=30, sticky="ew")

        # Create a frame for other controls
        second_frame = customtkinter.CTkFrame(button_frame)
        second_frame.grid(row=7, column=0, pady=20, sticky="ew")
    
        # Zoom controls (Zoom In, Zoom Out)
        self.zoom_in_button = customtkinter.CTkButton(second_frame, text="Zoom In", font=(self.font_size,self.font_size), fg_color='gray', hover_color="dark gray", command=self.zoom_in)
        self.zoom_out_button = customtkinter.CTkButton(second_frame, text="Zoom Out", font=(self.font_size,self.font_size), fg_color='gray', hover_color="dark gray", command=self.zoom_out)
        # Arrange zoom buttons horizontally
        self.zoom_in_button.grid(row=1, column=0,ipady=12, padx=30, pady=10,sticky="ew")
        self.zoom_out_button.grid(row=1, column=1, ipady=12, padx=30, pady=10,sticky="ew")

        #Number of segmentation points
        self.add_points_button = customtkinter.CTkButton(second_frame, text="Add points", font=(self.font_size,self.font_size), fg_color='gray', hover_color="dark gray", command=self.add_points_for_segmentation)
        self.reduce_points_button = customtkinter.CTkButton(second_frame, text="Remove points", font=(self.font_size,self.font_size), fg_color='gray', hover_color="dark gray", command=self.reduce_points_for_segmentation)
        # Arrange  buttons horizontally
        self.add_points_button.grid(row=2, column=0,ipady=12, padx=30, pady=10,sticky="ew")
        self.reduce_points_button.grid(row=2, column=1, ipady=12, padx=30, pady=10,sticky="ew")


        # shortcuts
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.bind("s", lambda event: self.save_image())
        self.root.bind("<Return>", lambda event: self.save_image())
        self.root.bind("r", lambda event: self.reset_rectangle())
        self.root.bind("p", lambda event: self.perform_segmentation())

        # for undo action
        self.prev_image_name = None


        #Create annotation 
        os.makedirs(FOLDER_ANNOTATED, exist_ok=True)
        os.makedirs(FOLDER_ORIGINAL_IMAGES, exist_ok=True)
        os.makedirs(FOLDER_MASKS, exist_ok=True)
        os.makedirs(FOLDER_ANNOTATIONS, exist_ok=True)
        
    # Function to display the current slider value
    def update_label(self, value):
        self.value_label.config(text=f"Value: {value}")

    def load_seg_model(self,model_path) -> ViT_seg:
        """Define arguments for MicroSegNet model initialization
        Args: model_path - path to the MUCSNet model

        Outputs:
            net[ViT_seg] - segmentation model
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
        parser.add_argument('--num_classes', type=int,default=1, help='output channel of network')
        parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
        parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
        parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
        args, unknown = parser.parse_known_args()


        #Define ViT model and load weights
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
        if args.vit_name.find('R50') !=-1:
            config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        #Model
        net.load_state_dict(torch.load(model_path))
        #Set model to eval mode
        net.eval()
        return net

    #Method that initializes images dir load 
    def load_images(self) -> None:
        """Load multiple images from a selected directory.
        
        Currently have the support for .jpeg, .jpg and .png images.
        """
        directory_path = customtkinter.filedialog.askdirectory(title="Select a directory containing images")
        if directory_path:
            # Filter for valid image files
            valid_extensions = {".jpeg", ".jpg", ".png"}
            #Store the image paths to the list
            self.image_paths = [
                os.path.join(directory_path, file)
                for file in os.listdir(directory_path)
                if os.path.splitext(file)[1].lower() in valid_extensions
            ]
            
            #Load image by image
            if self.image_paths:
                self.image_paths=natsorted(self.image_paths)
                self.current_image_index = 0
                #self.annotated_image_conunter=0
                self.load_current_image()
            else:
                print("No valid image files found in the selected directory.")

    #Method that loads the single image file
    def load_current_image(self) -> None:
        """Load the image based on the current_image_index."""
        torch.cuda.empty_cache()

        if self.current_image_index < len(self.image_paths):
            #Store file path, name and dataset number
            file_path=self.image_paths[self.current_image_index]
            annotated_file_paths=os.listdir(FOLDER_ORIGINAL_IMAGES)
            annotated_image_names=[]
            for annotated_file_path in annotated_file_paths:

                annotated_dataset_number=annotated_file_path.split("_")[-4]
                annotated_image_counter=annotated_file_path.split("_")[-1].split(".p")[0]

                annotated_image_names.append(annotated_dataset_number+"_"+annotated_image_counter)



            self.file_name=str(file_path.split("/")[-1])
            self.dataset_number=str(file_path.split("_")[-3])
            self.image_counter=str(file_path.split("_")[-2])
            #Combined image name for sorting purposes
            self.image_name=self.dataset_number+"_"+self.image_counter

            if annotated_image_names:
                if str(self.image_name) not in annotated_image_names:
                        #Define names for stored original (img) images and masks (gt)
                        self.original_image_name=f"microUS_{self.dataset_number}_img_slice_{self.image_counter}"
                        self.mask_image_name=f"microUS_{self.dataset_number}_gt_slice_{self.image_counter}"
                        #self.annotated_image_conunter+=1
                        #Set the canvas title
                        self.root.title(self.original_image_name)

                        if file_path:
                            #Load image
                            self.operational_image=cv2.imread(file_path)
                            self.operational_image=cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)
                            #Store the original image shape
                            self.image_shape=[self.operational_image.shape[1],self.operational_image.shape[0]] #width, height
                            #Copy the original image of original shape
                            self.original_image=self.operational_image.copy()
                            #Starting zoom value
                            self.zoom_value=1.0
                            self.update_canvas()
                            self.segmentation_performed=False
                            #Inintialize the empty mask
                            self.empty_mask = []
                            #Perform the initial segmentaion using MUCSNet
                            self.perform_segmentation()

                            #Setup the mask used for polygon drawing
                            self.mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype=np.uint8)
                            self.drawing_polygon = False
                            self.polygon_points.clear()
                else:
                    self.current_image_index+=1
                    self.load_next_image()
                    pass
         
                
            else:
                #Define names for stored original (img) images and masks (gt)
                    self.original_image_name=f"microUS_{self.dataset_number}_img_slice_{self.image_counter}"
                    self.mask_image_name=f"microUS_{self.dataset_number}_gt_slice_{self.image_counter}"
                    #self.annotated_image_conunter+=1
                    #Set the canvas title
                    self.root.title(self.original_image_name)

                    if file_path:
                        #Load image
                        self.operational_image=cv2.imread(file_path)
                        self.operational_image=cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)
                        #Store the original image shape
                        self.image_shape=[self.operational_image.shape[1],self.operational_image.shape[0]] #width, height
                        #Copy the original image of original shape
                        self.original_image=self.operational_image.copy()
                        #Starting zoom value
                        self.zoom_value=1.0
                        self.update_canvas()
                        self.segmentation_performed=False
                        #Inintialize the empty mask
                        self.empty_mask = []
                        #Perform the initial segmentaion using MUCSNet
                        self.perform_segmentation()

                        #Setup the mask used for polygon drawing
                        self.mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype=np.uint8)
                        self.drawing_polygon = False
                        self.polygon_points.clear()

               
        else:
            self.clear_all_images()
            messagebox.showwarning("Annotation info.","There is no more images to annotate!")


    def del_prev_image(self) -> None:
        '''
        Delete previously annotated image if fix is needed.
        '''

        if self.prev_image_name == None:
            return
        
        os.remove(f'{FOLDER_ANNOTATIONS}/{self.prev_image_name}.png')
        os.remove(f'{FOLDER_ORIGINAL_IMAGES}/{self.prev_image_name}.png')
        prev_mask_name = self.prev_image_name.replace('img', 'gt')
        os.remove(f'{FOLDER_MASKS}/{prev_mask_name}.png')

        self.prev_image_name = None

        self.current_image_index -= 1
        self.load_current_image()


    def load_next_image(self):
        self.load_current_image()
    
    #Method that clears the annoator if there is no more images to annoatate
    def clear_all_images(self) -> None:
        """Clear all images and reset variables when there are no more images to process."""

        self.operational_image = None
        self.original_image = None
        self.annotated_image_real_size = None
        self.mask = None
        self.file_name = None
        self.image_paths = []
        self.current_image_index = 0
        self.zoom_value = 1.0
        # Clear the canvas or update the GUI accordingly
        self.canvas.delete("all")
        # Reset GUI window title or provide feedback
        self.root.title("No Images Loaded")

    #Action pefromed after click
    def on_click(self, event):
        if self.segmentation_performed:
            for i, (x,y) in enumerate(self.segment.contour_points):
                if abs((x+self.x) - event.x) < 5 and abs((y+self.y) - event.y) < 5:
                    self.selected_point=i
                    break
                else:
                    self.selected_point=None

    #Action performed while dragging
    def on_drag(self, event):
        if hasattr(self, "selected_point"):
            if self.selected_point is not None:
                self.segment.contour_points[self.selected_point]=[event.x-self.x, event.y-self.y]
                self.draw_contour()
            
    #Zoom in method
    def zoom_in(self) -> None:
        """Zoom in by increasing the zoom factor."""
        self.zoom_value = min(self.zoom_value + self.zoom_factor, self.max_zoom)
        self.update_canvas()

    #Zoom out method
    def zoom_out(self) -> None:
        """Zoom out by decreasing the zoom factor."""
        self.zoom_value = max(self.zoom_value - self.zoom_factor, self.min_zoom)
        self.update_canvas()

    #Start drawing a polygon
    def start_polygon_drawing(self) -> None:
        """Start polygon drawing mode."""
        self.drawing_polygon = True
        self.polygon_points.clear()
        self.segment = None
        if self.ready_for_first_polygon:
            messagebox.showinfo("Polygon mode", "Click on the canvas to add vertices. Double-click to complete.")
            #self.file_name=simpledialog.askstring("Polygon Mode", "Click on the canvas to add vertices. Double-click to complete. \n Enter the filename (without extension):")
            self.ready_for_first_polygon=False
            self.canvas.bind("<Button-1>", self.on_mouse_down)
            self.canvas.bind("<Double-1>", self.on_double_click) 
      
      

    #Mouse action methods:
    def on_mouse_down(self, event) -> None:
        if self.operational_image is not None:
            x, y = int((event.x - self.x) / self.zoom_value), int((event.y - self.y) / self.zoom_value)
            self.polygon_points.append((x, y))
            self.update_canvas()

    #Compplete the polygon on double click
    def on_double_click(self, event) -> None:
        """Complete the polygon when double-clicked."""
        self.number_of_polygons=1
        if self.drawing_polygon:
            self.complete_polygon()
            self.polygon=Polygon_Segmentator(self.zoomed_image, 
                                                self.file_name, 
                                                self.image_shape, 
                                                self.polygon_points, 
                                                self.mask)
            self.polygon.create_polygon()

    #Complete a polygon creation
    def complete_polygon(self) -> None:
        """Complete the polygon and stop polygon drawing mode."""
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Polygon Error", "At least 3 points are needed to complete a polygon.")
            return
        messagebox.showinfo("Polygon", "Polygon created successfully.")

        self.drawing_polygon = False
        cv2.polylines(self.operational_image, [np.array(self.polygon_points)], isClosed=True, color=(255, 255, 255), thickness=2)
        self.update_canvas()

                
                
    #Add points for segmentation
    def add_points_for_segmentation(self):
        self.points_for_segmentation+=10
        self.perform_segmentation()

    #Reduce points for segmentation
    def reduce_points_for_segmentation(self):
        if self.points_for_segmentation > 20:
            self.points_for_segmentation-=10
        self.perform_segmentation()
    
    #Empty segmentation for case the input image does not show object
    def perform_empty_mask_segmentation(self)->None:
        if self.operational_image is not None:
            self.empty_mask=np.zeros((self.operational_image.shape[0], self.operational_image.shape[1]), dtype=np.uint8)
    
    #Method that performs image segmentation
    def perform_segmentation(self)-> None :
        torch.cuda.empty_cache()

        if self.operational_image is not None:
            #Store the segmentation 
            self.segment = MUCSNet_Segmentator(self.zoomed_image,
                                               self.file_name,
                                               self.image_shape, 
                                               self.net,
                                               self.points_for_segmentation)
            #Setup the segmentation performed flag
            self.segmentation_performed=True
            #Replace the empty mask with the empty prediction if there are no contour points, else draw a contour on the canvas
            if self.segment.contour_points is None:
                self.empty_mask=self.segment.prediction
            else:
                self.draw_contour()

    #Draw the contour on the loaded image
    def draw_contour(self):
        self.canvas.delete("all")
        
        # Resize the image based on the zoom factor
        self.zoomed_width = int(self.operational_image.shape[1] * self.zoom_value)
        self.zoomed_height = int(self.operational_image.shape[0] * self.zoom_value)
        self.zoomed_image = cv2.resize(self.operational_image, (self.zoomed_width, self.zoomed_height))

        # Display image
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(self.zoomed_image))
        
        # Calculate coordinates to center the image
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.x = (canvas_width - self.zoomed_width) // 2
        self.y = (canvas_height - self.zoomed_height) // 2
        
        # Display the image at central coordinates
        self.canvas.create_image(self.x, self.y, anchor="nw", image=self.tk_image)

        if self.segment.contour_points is not None:
            for i, (x, y) in enumerate(self.segment.contour_points):
                # Scale the contour points based on the zoom factor
                x = int(x )
                y = int(y )
                # Offset the points to align with the centered image
                x += self.x
                y += self.y

                # Draw lines between consecutive points
                line_width=6
                prev_x = int(self.segment.contour_points[i - 1][0] ) + self.x
                prev_y = int(self.segment.contour_points[i - 1][1] ) + self.y
                self.canvas.create_line(prev_x, prev_y, x, y, width=line_width, fill="red")
                


        if self.segment.contour_points is not None:
            for i, (x, y) in enumerate(self.segment.contour_points):
                # Scale the contour points based on the zoom factor
                x = int(x )
                y = int(y )
                
                # Offset the points to align with the centered image
                x += self.x
                y += self.y
                # Draw points
                cirlce_radius=8
                self.canvas.create_oval(x - cirlce_radius, y - cirlce_radius, x + cirlce_radius, y + cirlce_radius, fill="blue", tags=f"point_{i}")
                
    #Update the canvas method
    def update_canvas(self, crosshair=None)-> None :
        if self.operational_image is not None:
            # Resize the image based on the zoom factor
            self.zoomed_width = int(self.operational_image.shape[1] * self.zoom_value)
            self.zoomed_height = int(self.operational_image.shape[0] * self.zoom_value)
            self.zoomed_image = cv2.resize(self.operational_image, (self.zoomed_width, self.zoomed_height))

            #Display image
            self.canvas.delete("all")
            self.tk_image=ImageTk.PhotoImage(image=Image.fromarray(self.zoomed_image))
            # Calculate coordinates to center the image
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.x = (canvas_width - self.zoomed_width) // 2
            self.y = (canvas_height - self.zoomed_height) // 2
            #Display the image at central coordinates
            self.canvas.create_image(self.x,self.y,anchor="nw", image=self.tk_image)

             # Draw temporary polygon while adding points
            if self.drawing_polygon==True and self.polygon_points:
                scaled_points = [(int(px * self.zoom_value) + self.x, int(py * self.zoom_value) + self.y) for px, py in self.polygon_points]
                for i in range(1, len(scaled_points)):
                        self.canvas.create_line(scaled_points[i - 1], scaled_points[i], fill="red", width=3)
                if len(scaled_points) > 1:
                        self.canvas.create_line(scaled_points[-1], scaled_points[0], fill="red", width=3)  # Close the loop


            #Display the cross for easier annotation
            if crosshair:
                cx,cy=crosshair
                canvas_width=self.canvas.winfo_width()
                canvas_height=self.canvas.winfo_height()
                self.canvas.create_line(0+self.x, cy+self.y, canvas_width+cx+self.x, cy+self.y, fill=COLOUR_LINE, dash=(2,2))
                self.canvas.create_line(cx+self.x, 0+self.y, cx+self.x, canvas_height+cy+self.y, fill=COLOUR_LINE, dash=(2,2))


    #Update canvas with annotated image
    def update_canvas_original_image(self) -> None:
        if self.original_image is not None:
            #Display image
            self.canvas.delete("all")
            # Resize the image based on the zoom factor
            zoomed_width = int(self.original_image.shape[1] * self.zoom_value)
            zoomed_height = int(self.original_image.shape[0] * self.zoom_value)
            self.zoomed_image = cv2.resize(self.original_image, (zoomed_width, zoomed_height))
            #Display image
            self.tk_image=ImageTk.PhotoImage(image=Image.fromarray(self.zoomed_image))
            self.canvas.create_image(self.x,self.y,anchor="nw", image=self.tk_image)
        
    #Reset the rectangle method (in case the user is not satisfied with the bounding box)
    def reset_rectangle(self) -> None:
        if self.operational_image is not None:
            # Reset the temporary image to the original
            self.operational_image=self.original_image.copy()
            #Update the canvas to the original image without annotations
            self.update_canvas_original_image()
            self.previous_segment = None
            #If polygon exists:
            
            #Stopped drawing polygons
            self.drawing_polygon = False
            #Set the environment ready for the first polygon
            self.ready_for_first_polygon=True
            #Set the environment ready for the fitst polygon edit
            self.ready_for_first_edit_polygon=True
            #Reset the segmentation mask to 0
            self.mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype=np.uint8)
            #Reset all the masks
            self.previous_mask=np.array([])
            # if self.query_box != None:
            #     self.query_box.destroy()

    #Save the image method
    def save_image(self) -> None:
        """Save the current image and move to next one."""
        if self.operational_image is not None:
           
            if len(self.empty_mask)>1:
                #Save empty mask
                mask_save_path=f"{FOLDER_MASKS}/{self.mask_image_name}.png"
                print(mask_save_path)
                cv2.imwrite(mask_save_path, self.empty_mask)
                self.empty_mask = []

                # Save the annotated image
                output_image_path=f"{FOLDER_ANNOTATIONS}/{self.original_image_name}.png"
                self.annotated_image_real_size= cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path, self.annotated_image_real_size)

                #Save original image
                output_image_path_original=f"{FOLDER_ORIGINAL_IMAGES}/{self.original_image_name}.png"
                self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path_original, self.original_image_rgb)



            elif self.segment != None:
                #Save mask
                mask_save_path=f"{FOLDER_MASKS}/{self.mask_image_name}.png"
                x_new, y_new = self.segment.contour_points[:, 0], self.segment.contour_points[:, 1]

                # Convert it back to the required format for OpenCV
                res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                self.smoothened_contours=[]
                self.smoothened_contours.append(np.asarray(res_array, dtype=np.int32))

                self.mask=np.zeros((self.operational_image.shape[0], self.operational_image.shape[1]), dtype=np.uint8)
                self.mask=cv2.drawContours(self.mask,self.smoothened_contours,0,(255,255,255),-1)

                cv2.imwrite(mask_save_path, self.mask)
                
                # Save the annotated image
                output_image_path=f"{FOLDER_ANNOTATIONS}/{self.original_image_name}.png"
                self.annotated_image_real_size=cv2.drawContours(self.operational_image,self.smoothened_contours,0,(255,255,255),2)
                cv2.imwrite(output_image_path, self.annotated_image_real_size)

                #Save original image
                output_image_path_original=f"{FOLDER_ORIGINAL_IMAGES}/{self.original_image_name}.png"
                self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path_original, self.original_image_rgb)
            
            else:
                #Save mask
                mask_save_path=f"{FOLDER_MASKS}/{self.mask_image_name}.png"
                cv2.imwrite(mask_save_path, (self.polygon.resized_mask * 255).astype(np.uint8))

                # Save the annotated image
                output_image_path=f"{FOLDER_ANNOTATIONS}/{self.original_image_name}.png"
                self.image1= cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path, self.image1)

                #Save original image
                output_image_path_original=f"{FOLDER_ORIGINAL_IMAGES}/{self.original_image_name}.png"
                self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path_original, self.original_image_rgb)
            
           

            #Reset the points coordinates     
            self.rect_start=None
            self.rect_end=None
        
            
            #Reset the segmentation mask to 0
            self.mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype=np.uint8)
            # Reset the operational image to the original
            self.operational_image=None
            self.original_image=None

            #Reset all the masks
            self.previous_mask=np.array([])

            #Empty the mask
            self.empty_mask = []
            self.previous_segment = None

            # store this as previous image
            self.prev_image_name = self.original_image_name
            
            # Move to the next image
            self.current_image_index += 1
            self.load_current_image()
            



if __name__=="__main__":
    root=customtkinter.CTk()
    
    # maximize
    # w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    # root.geometry("%dx%d+0+0" % (w, h))
    
    app=ContourEditor(root)
    root.mainloop()