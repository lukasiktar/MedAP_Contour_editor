import os
import sys
import json
import cv2
import torch
import argparse
import numpy as np
from PIL import Image, ImageTk
from constants import *
import customtkinter
import datetime

from natsort import natsorted
from tkinter import Canvas, messagebox
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from MUCSNet_Segment import MUCSNet_Segmentator
from Polygon_segmentator import Polygon_Segmentator

sys.setrecursionlimit(10000) 

def annotator_menu_callback(choice):
    print(f'new annotator: {choice}')

def load_stats(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"total_segmented": 0, "last_update": None, "total_sessions": 0}

def get_total_segmented(filepath):
    stats = load_stats(filepath)
    return stats['total_segmented']

def save_stats(filepath, stats):
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)

def update_stats_field(filepath, key, value):
    data = load_stats(filepath)
    data[key] = value
    data['last_update'] = datetime.datetime.now().isoformat()
    save_stats(filepath, data)
      
def increment_segmented(filepath, count=1):
    stats = load_stats(filepath)
    stats['total_segmented'] += count
    stats['last_update'] = datetime.datetime.now().isoformat()
    save_stats(filepath, stats)

class ContourEditor:
    def __init__(self, root: customtkinter.CTk):

        #Segmentation model load
        MODEL_PATH ="epoch_19.pth"
        self.net = self.load_seg_model(MODEL_PATH)
        #Root setup
        self.root=root
        self.root.title("MedAP Contour Editor")
        self.root.configure(bg=COLOUR_ROOT_BG)
        customtkinter.set_appearance_mode("dark")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Initialize variables
        self.operational_image=None #Operational image
        self.original_image=None    #Original image
        self.tk_image=None          #Image format for canvas
        self.preannotated_mask=[]

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
        
        self.root.attributes('-zoomed', True)

        #Create GUI elements
        self.canvas = Canvas(root, bg=COLOUR_CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=0, pady=0)


        # Create a frame for the buttons on the right side
        button_frame =customtkinter.CTkFrame(root)
        button_frame.pack(side="right", fill="y", padx=30)

        #Buttons
        self.load_button = customtkinter.CTkButton(button_frame,text="Load Dataset", font=(self.font_size,self.font_size), command=self.load_images)          
        self.save_button = customtkinter.CTkButton(button_frame, text="Save Annotation (Enter)", font=(self.font_size,self.font_size), fg_color='green', hover_color="dark green", command=self.save_image)
        self.reset_button = customtkinter.CTkButton(button_frame, text="Reset Annotation (R)", font=(self.font_size,self.font_size), command=self.reset_rectangle)
        self.draw_polygon_button = customtkinter.CTkButton(button_frame, text="Draw Polygon", font=(self.font_size,self.font_size), command=self.start_polygon_drawing)
        self.perform_segmentation_button = customtkinter.CTkButton(button_frame, text="Perform segmentation (P)", font=(self.font_size,self.font_size), command=self.perform_segmentation)
        self.draw_empty_segmetation_button=customtkinter.CTkButton(button_frame, text="Empty Segmentation", font=(self.font_size,self.font_size), command=self.perform_empty_mask_segmentation)
        self.exit_button = customtkinter.CTkButton(button_frame, text="Exit MedAP", font=(self.font_size,self.font_size), fg_color='red', hover_color="dark red", command=root.quit)

        self.undo_button = customtkinter.CTkButton(button_frame, text="Fix previous", font=(self.font_size,self.font_size), fg_color='medium slate blue', hover_color="dark slate blue", command=self.del_prev_image)
        self.annotator_dropdown = customtkinter.CTkOptionMenu(button_frame, values=DOCTORS_OPTIONS, command=annotator_menu_callback)
        self.interesting_checkbox_value = False
        self.interesting_checkbox = customtkinter.CTkCheckBox(button_frame, text='Interesting')

        # Arrange these buttons in the grid (1 column, multiple rows)
        self.load_button.grid(row=0, column=0, ipadx=12, ipady=12, padx=20, pady=10,sticky="ew")
        self.save_button.grid(row=1, column=0, ipadx=12, ipady=12, padx=20, pady=20,sticky="ew")
        self.reset_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.draw_polygon_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.perform_segmentation_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.draw_empty_segmetation_button.grid(row=5, column=0, padx=20, pady=20, sticky="ew")
        self.exit_button.grid(row=6, column=0, ipadx=12, ipady=12, padx=20, pady=30, sticky="ew")
        self.undo_button.grid(row=8, column=0, ipadx=0, ipady=12, padx=20, pady=30, sticky="ew")

        self.annotator_dropdown.grid(row=9, column=0, ipadx=0, ipady=12, padx=20, pady=30, sticky="ew")
        self.interesting_checkbox.grid(row=10, column=0, ipadx=0, ipady=12, padx=20, pady=30, sticky="ew")

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
        os.makedirs(FOLDER_PREMASKS, exist_ok=True)
        os.makedirs(FOLDER_INFORMATION, exist_ok=True)

        # keep track of total number of session
        stats = load_stats(STATS_FILENAME)
        total_sessions = stats['total_sessions']
        update_stats_field(STATS_FILENAME, 'total_sessions', total_sessions+1)
        
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
        #self.directory_path = customtkinter.filedialog.askdirectory(title="Select a directory containing images")
        self.directory_path = FOLDER_DATA
        self.mask_directory_path=f"{self.directory_path}_mask"
        os.makedirs(self.mask_directory_path,exist_ok=True)
        self.preannotation_mask_directory_path=f"{self.directory_path}_premask"
        os.makedirs(self.preannotation_mask_directory_path,exist_ok=True)
        if self.directory_path:
            # Filter for valid image files
            valid_extensions = {".jpeg", ".jpg", ".png", ".dcm"}
            #Store the image paths to the list
            self.image_paths = [
                os.path.join(self.directory_path, file)
                for file in os.listdir(self.directory_path)
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
            self.file_path=self.image_paths[self.current_image_index]
            
            annotated_file_paths=os.listdir(FOLDER_ORIGINAL_IMAGES)
            annotated_image_names=[]
            for annotated_file_path in annotated_file_paths:
                #print(annotated_file_path)
                annotated_dataset_number=annotated_file_path.split("_")[-2]
                annotated_image_counter=annotated_file_path.split("_")[-1].split(".p")[0]
                #print(f"annoatetated dataset {annotated_dataset_number}" )
                #print(f"annotated counter {annotated_image_counter}")

                annotated_image_names.append(annotated_dataset_number+"_"+annotated_image_counter)


            #print(self.file_path)
            self.file_name=str(self.file_path.split("/")[-1])
            self.dataset_number=str(self.file_path.split("_")[-2])
            self.image_counter=str(self.file_path.split("_")[-1].split('.d')[0])
            #Combined image name for sorting purposes
            #self.image_name=self.dataset_number+"_"+self.image_counter
            self.image_name=str(self.file_name.split(".")[0])
            #print(self.image_name)
            if annotated_image_names:
                if str(self.image_name) not in annotated_image_names:
                        #Define names for stored original (img) images and masks (gt)
                        self.original_image_name=f"{self.dataset_number}_img_slice_{self.image_counter}"
                        self.mask_image_name=f"{self.dataset_number}_gt_slice_{self.image_counter}"
                        #self.annotated_image_conunter+=1
                        #Set the canvas title
                        # self.root.title(self.original_image_name)
                        self.root.title(f'Image {get_total_segmented(STATS_FILENAME)+1}/TODO')
                      
                        if self.file_path:
                            path=self.file_path.split(".")[-1]
                            # print(path)
                            if self.file_path.split(".")[-1]=="dcm":
                                self.image_counter=self.file_path.split("_")[-1].split(".")[0]
                                # print(self.image_counter)
                                self.dicom_image_data=pydicom.dcmread(self.file_path)
                                image_data=self.dicom_image_data.pixel_array
                                self.operational_image=cv2.normalize(image_data, None, 0,255, cv2.NORM_MINMAX)
                                self.operational_image=cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)

                                #Store the original image shape
                                self.image_shape=[self.operational_image.shape[1],self.operational_image.shape[0]] #width, height
                                #Copy the original image of original shape
                                self.original_image=self.operational_image.copy()
                                #Starting zoom value
                                #self.zoom_value=1.0
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
                                #Load image
                                self.operational_image=cv2.imread(self.file_path)
                                self.operational_image=cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)
                                #Store the original image shape
                                self.image_shape=[self.operational_image.shape[1],self.operational_image.shape[0]] #width, height
                                #Copy the original image of original shape
                                self.original_image=self.operational_image.copy()
                                #Starting zoom value
                                #self.zoom_value=1.0
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
                    self.original_image_name=f"{self.dataset_number}_img_slice_{self.image_counter}"
                    self.mask_image_name=f"{self.dataset_number}_gt_slice_{self.image_counter}"
                    #self.annotated_image_conunter+=1
                    #Set the canvas title
                    self.root.title(self.original_image_name)

                    if self.file_path:
                        if self.file_path.split(".")[-1]=="dcm":
                                self.image_counter=self.file_path.split("_")[-1].split(".")[0]
                                # print(self.image_counter)
                                self.dicom_image_data=pydicom.dcmread(self.file_path)
                                image_data=self.dicom_image_data.pixel_array
                                self.operational_image=cv2.normalize(image_data, None, 0,255, cv2.NORM_MINMAX)
                                self.operational_image=cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)

                                #Store the original image shape
                                self.image_shape=[self.operational_image.shape[1],self.operational_image.shape[0]] #width, height
                                #Copy the original image of original shape
                                self.original_image=self.operational_image.copy()
                                #Starting zoom value
                                #self.zoom_value=1.0
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
                            #Load image
                            self.operational_image=cv2.imread(self.file_path)
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
        try:
            print(f'will remove {FOLDER_INFORMATION}/{self.prev_image_name}.txt')
            os.remove(f'{FOLDER_INFORMATION}/{self.prev_image_name}.txt')
            os.remove(f'{FOLDER_ANNOTATIONS}/{self.prev_image_name}.png')
            os.remove(f'{FOLDER_ORIGINAL_IMAGES}/{self.prev_image_name}.png')
            prev_mask_name = self.prev_image_name.replace('img', 'gt')
            # os.remove(f'{FOLDER_MASKS}/{prev_mask_name}.png')
            # os.remove(f'{FOLDER_PREMASKS}/{prev_mask_name}.png')
        except:
            pass

        try:
            os.remove(f'{FOLDER_ANNOTATIONS}/{self.prev_image_name}.png')
            os.remove(f'{FOLDER_ORIGINAL_IMAGES}/{self.prev_image_name}.png')
            prev_mask_name = self.prev_image_name.replace('img', 'gt')
            os.remove(f'{FOLDER_MASKS}/{prev_mask_name}.png')
        except:
            pass
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

    #Action peformed after click
    def on_click(self, event):
        print(f"x,y : {event.x}, {event.y}")
        if self.segmentation_performed:
            for i, (x,y) in enumerate(self.segment.contour_points):
                if abs((x+self.x) - event.x) < 15 and abs((y+self.y) - event.y) < 15:
                    self.selected_point=i
                    break
                else:
                    self.selected_point=None
        else:
            for i, (x,y) in enumerate(self.scaled_polygon_points):
               
                if abs((x+self.x) - event.x) < 15 and abs((y+self.y) - event.y) < 15:
                    self.selected_point=i
                    break
                else:
                    self.selected_point=None

    #Action performed while dragging
    def on_drag(self, event):
        if hasattr(self, "selected_point"):
            if self.segmentation_performed:
                if self.selected_point is not None:
                    self.segment.contour_points[self.selected_point]=[event.x-self.x, event.y-self.y]
                    self.draw_contour()
            else:
                if self.selected_point is not None:
                    self.polygon_points[self.selected_point]=[(event.x-self.x)/self.zoom_value, (event.y-self.y)/self.zoom_value]
                    self.draw_contour_polygon()
            
    #Zoom in method
    def zoom_in(self) -> None:
        """Zoom in by increasing the zoom factor."""
        self.zoom_value = min(self.zoom_value + self.zoom_factor, self.max_zoom)
        self.update_canvas()
        self.perform_segmentation()

    #Zoom out method
    def zoom_out(self) -> None:
        """Zoom out by decreasing the zoom factor."""
        self.zoom_value = max(self.zoom_value - self.zoom_factor, self.min_zoom)
        self.update_canvas()
        self.perform_segmentation()

    #Start drawing a polygon
    def start_polygon_drawing(self) -> None:
        """Start polygon drawing mode."""
        self.reset_rectangle()
        self.drawing_polygon = True
        self.polygon_points.clear()
        self.segment = None
        if self.ready_for_first_polygon:
            messagebox.showinfo("Polygon mode", "Click on the canvas to add vertices. Right click mouse to complete.")
            #self.file_name=simpledialog.askstring("Polygon Mode", "Click on the canvas to add vertices. Double-click to complete. \n Enter the filename (without extension):")
            self.ready_for_first_polygon=False
            self.canvas.bind("<Button-1>", self.on_mouse_down)
            self.canvas.bind("<Button-3>", self.on_double_click) 
      
      

    #Mouse action methods:
    def on_mouse_down(self, event) -> None:
        if self.operational_image is not None:
            x, y = int((event.x - self.x) / self.zoom_value), int((event.y - self.y) / self.zoom_value)
            self.polygon_points.append((x, y))
            self.update_canvas()

    #Compplete the polygon on double click
    def on_double_click(self, event) -> None:
        """Complete the polygon when double-clicked."""
        # TODO change name of function, called when right click is pressed
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
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<Double-1>")
        self.canvas.bind("<Button-1>", self.on_click)
        messagebox.showinfo("Polygon", "Polygon created successfully.")

        self.drawing_polygon = False
        self.segmentation_performed=False
        #cv2.polylines(self.operational_image, [np.array(self.polygon_points)], isClosed=True, color=(255, 255, 255), thickness=2)
        self.update_canvas()
        self.draw_contour_polygon()

                
                
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
            self.empty_mask=np.zeros((self.original_image.shape[0], self.original_image.shape[1]), dtype=np.uint8)
            self.preannotated_mask=np.zeros((self.original_image.shape[0], self.original_image.shape[1]), dtype=np.uint8)
    
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
                #Save mask
                x_new, y_new = self.segment.contour_points[:, 0], self.segment.contour_points[:, 1]

                # Convert it back to the required format for OpenCV
                res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                self.smoothened_contours=[]
                self.smoothened_contours.append(np.asarray(res_array, dtype=np.int32))

                # Scale contours to original image size
                self.scaled_contours = []
                for contour in self.smoothened_contours:
                    contour = contour.astype(np.float32)

                    contour[:, 0, 0] /= self.zoom_value
                    contour[:, 0, 1]  /= self.zoom_value

                    self.scaled_contours.append(contour.astype(np.int32))

                self.preannotated_mask=np.zeros((self.operational_image.shape[0], self.operational_image.shape[1]), dtype=np.uint8)
                self.preannotated_mask=cv2.drawContours(self.preannotated_mask,self.scaled_contours,0,(255,255,255),-1)

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

        if self.segmentation_performed:
            if self.segment.contour_points is not None:
                for i, (x, y) in enumerate(self.segment.contour_points):
                    # Scale the contour points based on the zoom factor
                    x = int(x )
                    y = int(y )
                    # Offset the points to align with the centered image
                    x += self.x
                    y += self.y

                    # Draw lines between consecutive points
                    line_width=3
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
                    cirlce_radius=4
                    self.canvas.create_oval(x - cirlce_radius, y - cirlce_radius, x + cirlce_radius, y + cirlce_radius, fill="blue", tags=f"point_{i}")
        else:
            if self.polygon_points is not None:
                for i, (x, y) in enumerate(self.polygon_points):
                    # Scale the contour points based on the zoom factor
                    x = int(x )*self.zoom_value
                    y = int(y )*self.zoom_value
                    # Offset the points to align with the centered image
                    x += self.x
                    y += self.y

                    # Draw lines between consecutive points
                    line_width=3
                    prev_x = int(self.polygon_points[i - 1][0] ) + self.x
                    prev_y = int(self.polygon_points[i - 1][1] ) + self.y
                    self.canvas.create_line(prev_x, prev_y, x, y, width=line_width, fill="red")
                    


            if self.polygon_points is not None:
                for i, (x, y) in enumerate(self.polygon_points):
                    # Scale the contour points based on the zoom factor
                    x = int(x )*self.zoom_value
                    y = int(y )*self.zoom_value
                    
                    # Offset the points to align with the centered image
                    x += self.x
                    y += self.y
                    # Draw points
                    cirlce_radius=4
                    self.canvas.create_oval(x - cirlce_radius, y - cirlce_radius, x + cirlce_radius, y + cirlce_radius, fill="blue", tags=f"point_{i}")

    #Display image with contours for polygon draw
    def draw_contour_polygon(self):
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

        self.scaled_polygon_points=[]
        if self.polygon_points is not None:
            for i, (x, y) in enumerate(self.polygon_points):
                # Scale the contour points based on the zoom factor
                x = int(x )*self.zoom_value
                y = int(y )*self.zoom_value
                self.scaled_polygon_points.append((x,y))
                # Offset the points to align with the centered image
                x += self.x
                y += self.y

                # Draw lines between consecutive points
                line_width=3
                prev_x = int(self.polygon_points[i - 1][0] )*self.zoom_value + self.x
                prev_y = int(self.polygon_points[i - 1][1] )*self.zoom_value + self.y
                self.canvas.create_line(prev_x, prev_y, x, y, width=line_width, fill="red")
                


        if self.polygon_points is not None:
            for i, (x, y) in enumerate(self.polygon_points):
                # Scale the contour points based on the zoom factor
                x = int(x )*self.zoom_value
                y = int(y )*self.zoom_value
                
                # Offset the points to align with the centered image
                x += self.x
                y += self.y
                # Draw points
                cirlce_radius=4
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
            
            self.segment = None
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

    def save_image_info(self, filepath) -> None:
        '''
        Save information about annotator, cancer possibility time of annotation and other important stuff.
        '''
        is_interesting = 'yes' if self.interesting_checkbox.get() else 'no'
        data = {
            "annotator": self.annotator_dropdown._current_value,
            "is_interesting": is_interesting,
            "time": datetime.datetime.now().isoformat()
        }

        with open(filepath, 'x') as f:
            json.dump(data, f, indent=2)
        

    #Save the image method
    def save_image(self) -> None:
        """Save the current image and move to next one."""

        if self.operational_image is None:
            return

        info_path = f"{FOLDER_INFORMATION}/{self.image_name}.txt"
        self.save_image_info(info_path)
        #mask_save_path=f"{FOLDER_PREMASKS}/{self.mask_image_name}.png"
        #cv2.imwrite(mask_save_path, self.preannotated_mask)
        image_name_dcm=self.mask_image_name.split("/")[-1]
        image_name_dcm=image_name_dcm.replace("_gt_slice_","_")
        preannotation_mask_directory_path=f"{self.preannotation_mask_directory_path}/{image_name_dcm}.dcm"

        png_preannotation_mask_directory_path=f"{FOLDER_PREMASKS}/{image_name_dcm}.png"
        if self.preannotated_mask is None or not isinstance(self.preannotated_mask, np.ndarray):
            # Create an empty mask with the same shape as the original image
            height, width = self.original_image.shape[:2]
            self.preannotated_mask = np.zeros((height, width), dtype=np.uint8)

        cv2.imwrite(png_preannotation_mask_directory_path, self.preannotated_mask)
                
        # Create file meta with original transfer syntax
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = self.dicom_image_data.file_meta.TransferSyntaxUID
        file_meta.MediaStorageSOPClassUID = self.dicom_image_data.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = self.dicom_image_data.file_meta.ImplementationClassUID

        # Create new dataset inheriting original metadata
        ds = FileDataset(preannotation_mask_directory_path, {}, file_meta=file_meta, preamble=self.dicom_image_data.preamble)


        # Copy all original metadata except pixel-related tags
        for elem in self.dicom_image_data:
            if elem.tag not in [0x7FE00010, 0x00280010, 0x00280011]:  # Skip PixelData, Rows, Columns
                ds.add(elem)
        
        # Set mask-specific attributes
        ds.Rows, ds.Columns = self.preannotated_mask.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsStored = self.dicom_image_data.BitsStored
        ds.BitsAllocated = self.dicom_image_data.BitsAllocated
        ds.HighBit = self.dicom_image_data.HighBit
        ds.PixelRepresentation = self.dicom_image_data.PixelRepresentation

        # Set mask pixel data (ensure correct dtype)
        ds.PixelData = self.preannotated_mask.astype(self.dicom_image_data.pixel_array.dtype).tobytes()
        
        # Update required UIDs and timestamps
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
        
        # Modify identification tags
        ds.SeriesDescription = "Segmentation Mask"
        #ds.SeriesNumber = str(int(self.dicom_image_data.SeriesNumber) + 1000) if hasattr(self.dicom_image_data, 'SeriesNumber') else "1000"
        
        # Set appropriate SOP Class (Secondary Capture)
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
        
        # Save the new DICOM file
        ds.save_as(preannotation_mask_directory_path)


        if len(self.empty_mask)>1:
                print("Empty mask")
                #Save empty mask
                mask_image_name=self.mask_image_name.split("/")[-1]
                mask_image_name=mask_image_name.replace("_gt_slice_","_")
                png_mask_save_path=f"{FOLDER_MASKS}/{mask_image_name}.png"
                cv2.imwrite(png_mask_save_path, self.empty_mask)

                #Original mask
                mask_save_path=f"{self.mask_directory_path}/{image_name_dcm}.dcm"

                # Create file meta with original transfer syntax
                file_meta = FileMetaDataset()
                file_meta.TransferSyntaxUID = self.dicom_image_data.file_meta.TransferSyntaxUID
                file_meta.MediaStorageSOPClassUID = self.dicom_image_data.SOPClassUID
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.ImplementationClassUID = self.dicom_image_data.file_meta.ImplementationClassUID

                # Create new dataset inheriting original metadata
                ds = FileDataset(mask_save_path, {}, file_meta=file_meta, preamble=self.dicom_image_data.preamble)


                # Copy all original metadata except pixel-related tags
                for elem in self.dicom_image_data:
                    if elem.tag not in [0x7FE00010, 0x00280010, 0x00280011]:  # Skip PixelData, Rows, Columns
                        ds.add(elem)
                
                # Set mask-specific attributes
                ds.Rows, ds.Columns = self.empty_mask.shape
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsStored = self.dicom_image_data.BitsStored
                ds.BitsAllocated = self.dicom_image_data.BitsAllocated
                ds.HighBit = self.dicom_image_data.HighBit
                ds.PixelRepresentation = self.dicom_image_data.PixelRepresentation

                # Set mask pixel data (ensure correct dtype)
                ds.PixelData = self.empty_mask.astype(self.dicom_image_data.pixel_array.dtype).tobytes()
                
                # Update required UIDs and timestamps
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
                ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
                
                # Modify identification tags
                ds.SeriesDescription = "Segmentation Mask"
                #ds.SeriesNumber = str(int(self.dicom_image_data.SeriesNumber) + 1000) if hasattr(self.dicom_image_data, 'SeriesNumber') else "1000"
                
                # Set appropriate SOP Class (Secondary Capture)
                ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
                
                # Save the new DICOM file
                ds.save_as(mask_save_path)

                #Preannotated mask
                premask_save_path=f"{self.mask_directory_path}/{image_name_dcm}.dcm"

                # Create file meta with original transfer syntax
                file_meta = FileMetaDataset()
                file_meta.TransferSyntaxUID = self.dicom_image_data.file_meta.TransferSyntaxUID
                file_meta.MediaStorageSOPClassUID = self.dicom_image_data.SOPClassUID
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.ImplementationClassUID = self.dicom_image_data.file_meta.ImplementationClassUID

                # Create new dataset inheriting original metadata
                ds = FileDataset(premask_save_path, {}, file_meta=file_meta, preamble=self.dicom_image_data.preamble)


                # Copy all original metadata except pixel-related tags
                for elem in self.dicom_image_data:
                    if elem.tag not in [0x7FE00010, 0x00280010, 0x00280011]:  # Skip PixelData, Rows, Columns
                        ds.add(elem)
                
                # Set mask-specific attributes
                ds.Rows, ds.Columns = self.empty_mask.shape
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsStored = self.dicom_image_data.BitsStored
                ds.BitsAllocated = self.dicom_image_data.BitsAllocated
                ds.HighBit = self.dicom_image_data.HighBit
                ds.PixelRepresentation = self.dicom_image_data.PixelRepresentation

                # Set mask pixel data (ensure correct dtype)
                ds.PixelData = self.empty_mask.astype(self.dicom_image_data.pixel_array.dtype).tobytes()
                
                # Update required UIDs and timestamps
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
                ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
                
                # Modify identification tags
                ds.SeriesDescription = "Segmentation Mask"
                #ds.SeriesNumber = str(int(self.dicom_image_data.SeriesNumber) + 1000) if hasattr(self.dicom_image_data, 'SeriesNumber') else "1000"
                
                # Set appropriate SOP Class (Secondary Capture)
                ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
                
                # Save the new DICOM file
                ds.save_as(premask_save_path)

                self.empty_mask = []

                # Save the annotated image
                original_image_name=self.original_image_name.split("/")[-1]
                original_image_name=original_image_name.replace("_img_slice_","_")
                output_image_path=f"{FOLDER_ANNOTATIONS}/{original_image_name}.png"
                self.annotated_image_real_size= cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path, self.annotated_image_real_size)

                #Save original image
                original_image_name=self.original_image_name.split("/")[-1]
                original_image_name=original_image_name.replace("_img_slice_","_")
                output_image_path_original=f"{FOLDER_ORIGINAL_IMAGES}/{original_image_name}.png"
                self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path_original, self.original_image_rgb)


        elif self.segment != None:
            if self.file_path.split(".")[-1]=="dcm":
                #Save mask
                image_name_dcm=self.mask_image_name.split("/")[-1]
                image_name_dcm=image_name_dcm.replace("_gt_slice_","_")
                mask_save_path=f"{self.mask_directory_path}/{image_name_dcm}.dcm"
                png_mask_save_path=f"{FOLDER_MASKS}/{image_name_dcm}.png"
                # print(mask_save_path)
                x_new, y_new = self.segment.contour_points[:, 0], self.segment.contour_points[:, 1]

                # Convert it back to the required format for OpenCV
                res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                self.smoothened_contours=[]
                self.smoothened_contours.append(np.asarray(res_array, dtype=np.int32))

                 # Scale contours to original image size
                self.scaled_contours = []
                for contour in self.smoothened_contours:
                    contour = contour.astype(np.float32)

                    contour[:, 0, 0] /= self.zoom_value
                    contour[:, 0, 1]  /= self.zoom_value

                    self.scaled_contours.append(contour.astype(np.int32))

                self.mask=np.zeros((self.operational_image.shape[0], self.operational_image.shape[1]), dtype=np.uint8)
                self.mask=cv2.drawContours(self.mask,self.scaled_contours,0,(255,255,255),-1)

                cv2.imwrite(png_mask_save_path, self.mask)


                # Create file meta with original transfer syntax
                file_meta = FileMetaDataset()
                file_meta.TransferSyntaxUID = self.dicom_image_data.file_meta.TransferSyntaxUID
                file_meta.MediaStorageSOPClassUID = self.dicom_image_data.SOPClassUID
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.ImplementationClassUID = self.dicom_image_data.file_meta.ImplementationClassUID

                # Create new dataset inheriting original metadata
                ds = FileDataset(mask_save_path, {}, file_meta=file_meta, preamble=self.dicom_image_data.preamble)


                # Copy all original metadata except pixel-related tags
                for elem in self.dicom_image_data:
                    if elem.tag not in [0x7FE00010, 0x00280010, 0x00280011]:  # Skip PixelData, Rows, Columns
                        ds.add(elem)
                
                # Set mask-specific attributes
                ds.Rows, ds.Columns = self.mask.shape
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsStored = self.dicom_image_data.BitsStored
                ds.BitsAllocated = self.dicom_image_data.BitsAllocated
                ds.HighBit = self.dicom_image_data.HighBit
                ds.PixelRepresentation = self.dicom_image_data.PixelRepresentation

                # Set mask pixel data (ensure correct dtype)
                ds.PixelData = self.mask.astype(self.dicom_image_data.pixel_array.dtype).tobytes()
                
                # Update required UIDs and timestamps
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
                ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
                
                # Modify identification tags
                ds.SeriesDescription = "Segmentation Mask"
                #ds.SeriesNumber = str(int(self.dicom_image_data.SeriesNumber) + 1000) if hasattr(self.dicom_image_data, 'SeriesNumber') else "1000"
                
                # Set appropriate SOP Class (Secondary Capture)
                ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
                
                # Save the new DICOM file
                ds.save_as(mask_save_path)

                #Save the original image
                #Save original image
                output_image_path_original=f"{FOLDER_ORIGINAL_IMAGES}/{image_name_dcm}.png"
                self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_image_path_original, self.original_image_rgb)




                # Save the annotated image
                output_image_path=f"{FOLDER_ANNOTATIONS}/{image_name_dcm}.png"
                self.annotated_image_real_size=cv2.drawContours(self.operational_image,self.scaled_contours,0,(255,255,255),2)
                cv2.imwrite(output_image_path, self.annotated_image_real_size)
            else:
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
            #if self.file_path.split(".")[-1]!="dcm":
                
            #Save mask
            mask_image_name=self.mask_image_name.replace("_gt_slice_","_")
            mask_image_name=mask_image_name.split("/")[-1]
            png_mask_save_path=f"{FOLDER_MASKS}/{mask_image_name}.png"
            mask_save_path=f"{self.mask_directory_path}/{image_name_dcm}.dcm"

            polygon_array = np.array(self.polygon_points)

            x_new, y_new = polygon_array[:, 0], polygon_array[:, 1]

            # Convert it back to the required format for OpenCV
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
            self.smoothened_contours=[]
            self.smoothened_contours.append(np.asarray(res_array, dtype=np.int32))

            #     # Scale contours to original image size
            # self.scaled_contours = []
            # for contour in self.smoothened_contours:
            #     contour = contour.astype(np.float32)

            #     contour[:, 0, 0] /= 1
            #     contour[:, 0, 1]  /= 1

            #     self.scaled_contours.append(contour.astype(np.int32))

            self.mask=np.zeros((self.original_image.shape[0], self.original_image.shape[1]), dtype=np.uint8)
            self.mask=cv2.drawContours(self.mask,self.smoothened_contours,0,(255,255,255),-1)

            cv2.imwrite(png_mask_save_path, self.mask)

            # Create file meta with original transfer syntax
            file_meta = FileMetaDataset()
            file_meta.TransferSyntaxUID = self.dicom_image_data.file_meta.TransferSyntaxUID
            file_meta.MediaStorageSOPClassUID = self.dicom_image_data.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.ImplementationClassUID = self.dicom_image_data.file_meta.ImplementationClassUID

            # Create new dataset inheriting original metadata
            ds = FileDataset(mask_save_path, {}, file_meta=file_meta, preamble=self.dicom_image_data.preamble)


            # Copy all original metadata except pixel-related tags
            for elem in self.dicom_image_data:
                if elem.tag not in [0x7FE00010, 0x00280010, 0x00280011]:  # Skip PixelData, Rows, Columns
                    ds.add(elem)
            
            # Set mask-specific attributes
            ds.Rows, ds.Columns = self.mask.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsStored = self.dicom_image_data.BitsStored
            ds.BitsAllocated = self.dicom_image_data.BitsAllocated
            ds.HighBit = self.dicom_image_data.HighBit
            ds.PixelRepresentation = self.dicom_image_data.PixelRepresentation

            # Set mask pixel data (ensure correct dtype)
            ds.PixelData = self.mask.astype(self.dicom_image_data.pixel_array.dtype).tobytes()
            
            # Update required UIDs and timestamps
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
            ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
            
            # Modify identification tags
            ds.SeriesDescription = "Segmentation Mask"
            #ds.SeriesNumber = str(int(self.dicom_image_data.SeriesNumber) + 1000) if hasattr(self.dicom_image_data, 'SeriesNumber') else "1000"
            
            # Set appropriate SOP Class (Secondary Capture)
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
            
            # Save the new DICOM file
            ds.save_as(mask_save_path)

            # Save the annotated image
            original_image_name=self.original_image_name.replace("_img_slice_","_")
            original_image_name=original_image_name.split("/")[-1]
            output_image_path=f"{FOLDER_ANNOTATIONS}/{original_image_name}.png"
            print(f"output image path: {output_image_path}")
            self.image1= cv2.cvtColor(self.operational_image, cv2.COLOR_BGR2RGB)
            self.image1=cv2.drawContours(self.image1,self.smoothened_contours,0,(255,255,255),2)
            cv2.imwrite(output_image_path, self.image1)

            #Save original image
            original_image_name=self.original_image_name.replace("_img_slice_","_")
            original_image_name=original_image_name.split("/")[-1]
            output_image_path_original=f"{FOLDER_ORIGINAL_IMAGES}/{original_image_name}.png"
            print(f"original image path: {output_image_path_original}")

            self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_image_path_original, self.original_image_rgb)
                
            
        increment_segmented('stats.json', count=1)

        #Reset the points coordinates     
        self.rect_start=None
        self.rect_end=None
    
        #Reset the segmentation mask to 0
        self.mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype=np.uint8)

        # Reset the operational image to the original
        self.operational_image=None
        self.original_image=None
        self.preannotated_mask=None
        

        #Reset all the masks
        self.previous_mask=np.array([])

        #Empty the mask
        self.empty_mask = []
        self.previous_segment = None

        # store this as previous image
        self.prev_image_name = self.image_name

        self.polygon_points.clear()
        self.selected_point=None
        self.segmentation_performed=False
        
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