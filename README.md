# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:

### Step1:

Import the required libraries.

### Step2:

Convert the image from BGR to RGB.

### Step3:

Apply the required filters for the image separately.

### Step4:

Plot the original and filtered image by using matplotlib.pyplot.

### Step5:

End the program.

## Program:
### Developed By   : HIRUTHIK SUDHAKAR
### Register Number: 212223240054
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```PYTHON

# expt-5(a)-smoothing filters - Average, Weighted Average, Gaussian and Median

import cv2
import numpy as np
from google.colab.patches import cv2_imshow


# Load the image
image = cv2.imread("/content/IMG-20230922-WA0002 (2).png")  # Replace with your actual image path
if image is None:
    raise ValueError("Image not found. Check the file path.")

# ------------------ 1. Averaging Filter ------------------
average_blur = cv2.blur(image, (5, 5))  # Kernel size (5x5)

# ------------------ 2. Weighted Averaging Filter ------------------
# Custom kernel (normalized)
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]], dtype=np.float32)
kernel /= np.sum(kernel)
weighted_blur = cv2.filter2D(image, -1, kernel)

# ------------------ 3. Gaussian Filter ------------------
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# ------------------ 4. Median Filter ------------------
median_blur = cv2.medianBlur(image, 5)  # Kernel size must be odd

# ------------------ Display Results ------------------
cv2_imshow(image)
cv2_imshow(average_blur)
cv2_imshow(weighted_blur)
cv2_imshow(gaussian_blur)
cv2_imshow(median_blur)
```

### 2. Sharpening Filters
```PYTHON
import cv2
import numpy as np

# Load the image
image = cv2.imread('../Desktop/ex01/egimg2.png')  # Replace with your actual image path
if image is None:
    raise ValueError("Image not found. Check the file path.")

# Convert to grayscale (for Laplacian Operator)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------ 1. Sharpening with Laplacian Linear Kernel ------------------
# Define a sharpening kernel using Laplacian
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32)

# Apply the kernel using filter2D
sharpened_kernel = cv2.filter2D(image, -1, laplacian_kernel)

# ------------------ 2. Sharpening with Laplacian Operator ------------------
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Convert grayscale laplacian to 3-channel to add back to color image
laplacian_color = cv2.merge([laplacian] * 3)

# Add Laplacian details to the original image to enhance edges
sharpened_operator = cv2.addWeighted(image, 1.0, laplacian_color, 1.0, 0)

# ------------------ Display Results ------------------
cv2.imshow("Original", image)
cv2.imshow(" Laplacian linear kernel", sharpened_kernel)
cv2.imshow(" Laplacian Operator", sharpened_operator)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
</br>
</br>
</br>
<img width="413" height="531" alt="image" src="https://github.com/user-attachments/assets/db53b60a-bded-4b36-b08e-3f3758e4fb00" />


</br>
</br>

ii)Using Weighted Averaging Filter
</br>
</br>
</br>
<img width="413" height="531" alt="image" src="https://github.com/user-attachments/assets/53b0ab4c-771b-4af5-8bf6-03ec7d4add53" />

</br>
</br>

iii)Using Gaussian Filter
</br>
</br>
</br>
<img width="413" height="531" alt="image" src="https://github.com/user-attachments/assets/2e581607-8b27-4020-ad7f-45987cdb2f00" />

</br>
</br>

iv) Using Median Filter
</br>
</br>
</br>
<img width="413" height="531" alt="image" src="https://github.com/user-attachments/assets/2506c590-e70c-47b7-8100-845df5d6f4ba" />

</br>
</br>

### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal
</br>
</br>
</br>
<img width="413" height="531" alt="image" src="https://github.com/user-attachments/assets/c1d34df0-5034-4e4f-965a-c25c04427860" />

</br>
</br>

ii) Using Laplacian Operator
</br>
</br>
</br>
<img width="413" height="531" alt="image" src="https://github.com/user-attachments/assets/48d0ff07-1cfb-4a00-ac8d-da6abe59b6f8" />

</br>
</br>

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
