import torch
import sys
import cv2
import numpy as np
from scipy.interpolate import splprep, splev



class MUCSNet_Segmentator:
    def __init__(self, zoomed_image, file_name, image_shape, net):

        """The initialization of MUCSNet_Segmentator
        The Outputs:  self.prediction - the segmentation mask
            self.contour_points - the points that construct the contour
            self.smoothened_contours - the smoothened contours from created mask
        """
        self.zoomed_image=zoomed_image
        self.filen_name=file_name
        self.image_shape=image_shape
        self.net=net
        
        patch_size=[224, 224]
        x, y = patch_size[0], patch_size[1]

        #Convert image to grayscale
        self.zoomed_image= cv2.cvtColor(self.zoomed_image, cv2.COLOR_BGR2GRAY)

        # Convert the image to a PyTorch tensor
        image_tensor = torch.from_numpy(self.zoomed_image).float()          # Convert to float32 tensor
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)   # Add batch and channel dimensions

        #Resize the tensor
        resized_image=cv2.resize(self.zoomed_image, (224,224))

        #Input to the neural network
        input_nn = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
                #Model outputs
                outputs, _, _, _, cls_output= net(input_nn)
                #Model predictions
                # Apply sigmoid to classification output
                cls_pred = torch.sigmoid(cls_output).squeeze()
                print(f"Confidence: {cls_pred}")
                #Check if the classification predicts an object 
                if cls_pred.item() < 0.9:  
                    # If no object is predicted, create a black mask
                    self.pred = np.zeros((self.zoomed_image.shape[0], self.zoomed_image.shape[1]), dtype=np.uint8)
                    self.contour_points=None
                else:
                    # If an object is predicted, proceed with segmentation as before
                    out = torch.sigmoid(outputs).squeeze()
                    self.pred = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    self.pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)
                
                #Create a binary mask from predicitons
                a = 1.0*(self.pred>0.5)
                self.prediction = a.astype(np.uint8)
                self.prediction = cv2.normalize(self.prediction, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.prediction=cv2.resize(self.prediction, (self.zoomed_image.shape[1],self.zoomed_image.shape[0]))
                
                #Find contours on predicted masks (used for visualization)
                contours, hierarchy = cv2.findContours(self.prediction,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #Smooth the contours
                self.smoothened_contours = []
                for contour in contours:
                    
                    x_1,y_1 = contour.T
                    # Convert from numpy arrays to normal arrays
                    x_1 = x_1.tolist()[0]
                    y_1 = y_1.tolist()[0]
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                    tck, u = splprep([x_1,y_1], u=None, s=0.0, k=1, per=1)
                    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                    u_new = np.linspace(u.min(), u.max(), 50)

                    
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                    x_new, y_new = splev(u_new, tck, der=0)

                    self.contour_points=np.column_stack((x_new, y_new))
                    # Convert it back to numpy format for opencv to be able to display it
                    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
                    self.smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
