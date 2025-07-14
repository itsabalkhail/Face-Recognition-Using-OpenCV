
import cv2
import numpy as np
import os
from PIL import Image

# Setup paths
known_faces_path = 'known_faces'  # Folder with known people
test_images_path = 'test_images'  # Folder with images to identify

# Storage for known faces
known_images = []
known_names = []

# Function to load image with PIL fallback for unsupported formats
def load_image_safe(image_path):
    """Load image with OpenCV first, fallback to PIL for unsupported formats"""
    try:
        # Try OpenCV first
        img = cv2.imread(image_path)
        if img is not None:
            return img
        
        # If OpenCV fails, try PIL
        print(f"OpenCV couldn't load {image_path}, trying PIL...")
        pil_image = Image.open(image_path)
        
        # Convert to RGB if it's RGBA
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'P':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL image to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return opencv_image
        
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# Check if known faces folder exists
if not os.path.exists(known_faces_path):
    print(f"Error: '{known_faces_path}' folder not found!")
    print("Please create a 'known_faces' folder and add images of people you want to recognize.")
    exit()

# Check if test images folder exists
if not os.path.exists(test_images_path):
    print(f"Error: '{test_images_path}' folder not found!")
    print("Please create a 'test_images' folder and add images you want to identify.")
    exit()

# Load known faces
print("Loading known faces...")
known_faces_list = os.listdir(known_faces_path)

# Updated to support more image formats
supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.avif')

for face_file in known_faces_list:
    if face_file.lower().endswith(supported_formats):
        face_img = load_image_safe(f'{known_faces_path}/{face_file}')
        if face_img is not None:
            known_images.append(face_img)
            # Remove file extension to get person name
            known_names.append(os.path.splitext(face_file)[0])
            print(f"Loaded known face: {face_file}")

if not known_images:
    print("No known faces found! Please add images to the 'known_faces' folder.")
    exit()

print(f"Known people: {known_names}")

# Function to detect faces using OpenCV
def findFaces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x+w, y+h, x))  # (top, right, bottom, left)
    
    return face_locations

# Function to extract face encodings
def getFaceEncoding(image):
    face_locations = findFaces(image)
    
    if face_locations:
        # Get the first face found
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        
        if face_image.size > 0:
            # Create simple encoding
            face_encoding = cv2.resize(face_image, (100, 100)).flatten()
            return face_encoding, (top, right, bottom, left)
    
    # If no face found, use full image
    resized = cv2.resize(image, (100, 100))
    return resized.flatten(), None

# Create encodings for known faces
print("Creating encodings for known faces...")
known_encodings = []
for i, img in enumerate(known_images):
    encoding, _ = getFaceEncoding(img)
    known_encodings.append(encoding)
    print(f"Encoded: {known_names[i]}")

print("Encoding complete!")

# Function to identify a person
def identifyPerson(test_image, tolerance=50000):
    # Get encoding for test image
    test_encoding, face_location = getFaceEncoding(test_image)
    
    # Compare with known faces
    best_match_index = -1
    min_distance = float('inf')
    
    for i, known_encoding in enumerate(known_encodings):
        distance = np.linalg.norm(test_encoding - known_encoding)
        if distance < min_distance:
            min_distance = distance
            best_match_index = i
    
    # Check if match is good enough
    if min_distance < tolerance:
        confidence = max(0, 100 - (min_distance / 1000))
        return known_names[best_match_index], confidence, face_location
    else:
        return "Unknown", 0, face_location

# Function to save image with results (optional)
def save_result_image(image, name, confidence, face_location, original_filename):
    """Save the result image with bounding box and label"""
    output_dir = 'output_results'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output filename
    base_name = os.path.splitext(original_filename)[0]
    output_filename = f"{base_name}_result.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")
    return output_path

# Function to display image properly
def display_image_robust(image, window_name, original_filename):
    """Display image with proper handling"""
    try:
        # Standard OpenCV display
        print(f"\nDisplaying image: {window_name}")
        print("Press any key to close the image window...")
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Resize if too large
        height, width = image.shape[:2]
        max_size = 800
        
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Display image
        cv2.imshow(window_name, image)
        
        # Wait for key press - improved method
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key != 255:  # Any key pressed
                break
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Clean up
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
        
        print("Image window closed.")
        
    except Exception as e:
        print(f"Error displaying image: {e}")
        print("Saving image instead...")
        # Only save if display fails
        save_result_image(image, "", 0, None, original_filename)

# Function to process a single image
def processImage(image_path):
    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print('='*50)
    
    # Load test image using safe loading function
    test_img = load_image_safe(image_path)
    if test_img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Create a copy for display
    display_img = test_img.copy()
    
    # Identify person
    name, confidence, face_location = identifyPerson(test_img)
    
    # Draw rectangle around face if found
    if face_location:
        top, right, bottom, left = face_location
        # Draw rectangle
        cv2.rectangle(display_img, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # Add text label
        if name != "Unknown":
            label = f"{name} ({confidence:.1f}%)"
            color = (0, 255, 0)  # Green for known person
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for unknown person
        
        # Add text background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(display_img, (left, top - text_height - 10), 
                     (left + text_width, top), color, -1)
        
        # Draw text
        cv2.putText(display_img, label, (left, top - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    # Print result
    print(f"Result: {name}", end="")
    if name != "Unknown":
        print(f" (Confidence: {confidence:.1f}%)")
    else:
        print()
    
    if face_location:
        print(f"Face detected at: top={face_location[0]}, right={face_location[1]}, bottom={face_location[2]}, left={face_location[3]}")
    else:
        print("No face detected in image")
    
    # Display image - no automatic saving
    window_name = f'Face Recognition Result - {os.path.basename(image_path)}'
    display_image_robust(display_img, window_name, os.path.basename(image_path))

# Process all test images
def processAllImages():
    test_images_list = os.listdir(test_images_path)
    image_files = [f for f in test_images_list if f.lower().endswith(supported_formats)]
    
    print(f"\nFound {len(image_files)} supported image files in test_images folder")
    
    if not image_files:
        print("No supported image files found!")
        return
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        image_path = f'{test_images_path}/{image_file}'
        processImage(image_path)
        
        # Ask user if they want to continue (except for last image)
        if i < len(image_files):
            while True:
                continue_choice = input("\nContinue to next image? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes', '']:
                    break
                elif continue_choice in ['n', 'no']:
                    print("Stopping processing.")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")

# Process specific image
def processSingleImage(image_name):
    image_path = f'{test_images_path}/{image_name}'
    if os.path.exists(image_path):
        processImage(image_path)
    else:
        print(f"Error: Image '{image_name}' not found in {test_images_path}")
        print("Available images:")
        available_images = [f for f in os.listdir(test_images_path) if f.lower().endswith(supported_formats)]
        for img in available_images:
            print(f"  - {img}")

# Function to test OpenCV display capability
def test_opencv_display():
    """Test if OpenCV can display images on this system"""
    print("\nTesting OpenCV display capability...")
    
    # Create a simple test image
    test_img = np.zeros((200, 300, 3), dtype=np.uint8)
    test_img[50:150, 100:200] = [0, 255, 0]  # Green rectangle
    
    try:
        cv2.imshow('Test Window', test_img)
        print("Test window should appear now. Press any key to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("OpenCV display test successful!")
        return True
    except Exception as e:
        print(f"OpenCV display test failed: {e}")
        print("Your system may not support OpenCV GUI features.")
        print("Results will be saved to 'output_results' folder instead.")
        return False

# Main menu
def main():
    print("\n" + "="*60)
    print("FACE RECOGNITION SYSTEM")
    print("="*60)
    print("This system will:")
    print("1. Load known faces from 'known_faces' folder")
    print("2. Compare test images from 'test_images' folder")
    print("3. Display results with bounding boxes and confidence scores")
    print("="*60)
    print("\nMenu Options:")
    print("1. Test OpenCV display capability")
    print("2. Process all images in test_images folder")
    print("3. Process specific image")
    print("4. Exit")
    print("="*60)
    print("Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP, AVIF")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            test_opencv_display()
        elif choice == '2':
            processAllImages()
        elif choice == '3':
            image_name = input("Enter image name (e.g., person1.jpg): ").strip()
            processSingleImage(image_name)
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")

# Run the program
if __name__ == "__main__":
    main()