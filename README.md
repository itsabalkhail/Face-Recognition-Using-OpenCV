# Face Recognition System

A simple yet robust face recognition system built with OpenCV and Python that can identify known people from test images.

## Features

- **Multi-format Support**: Supports PNG, JPG, JPEG, BMP, TIFF, WebP, and AVIF image formats
- **Robust Image Loading**: Uses OpenCV with PIL fallback for unsupported formats
- **Face Detection**: Automatically detects faces in images using Haar Cascade classifiers
- **Visual Results**: Displays results with bounding boxes and confidence scores
- **Batch Processing**: Process all images at once or individual images
- **Error Handling**: Comprehensive error handling for various scenarios
- **Cross-platform**: Works on Windows, macOS, and Linux

## Requirements

### Dependencies
```bash
pip install opencv-python
pip install numpy
pip install Pillow
```

### System Requirements
- Python 3.6 or higher
- OpenCV-compatible display system (for GUI mode)

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy Pillow
   ```
3. Create the required folder structure (see below)

## Folder Structure

Create the following folders in your project directory:

```
project_folder/
├── face_recognition.py
├── known_faces/          # Place known people's photos here
│   ├── john_doe.jpg
│   ├── jane_smith.png
│   └── bob_wilson.jpeg
├── test_images/          # Place images to identify here
│   ├── group_photo.jpg
│   ├── unknown_person.png
│   └── meeting.jpeg
└── output_results/       # Auto-created for saved results
```

## Usage

### Setup
1. **Add Known Faces**: Place photos of people you want to recognize in the `known_faces` folder
   - Use the person's name as the filename (e.g., `john_doe.jpg`)
   - One face per image works best
   - The system will use the filename (without extension) as the person's name

2. **Add Test Images**: Place images you want to analyze in the `test_images` folder
   - Can contain multiple faces
   - Various formats supported

### Running the System

Run the script:
```bash
python face_recognition.py
```

The system will present a menu with the following options:

#### 1. Test OpenCV Display Capability
Tests if your system can display OpenCV windows. If not, results will be saved to files instead.

#### 2. Process All Images
Processes all images in the `test_images` folder sequentially. You can choose to continue or stop between images.

#### 3. Process Specific Image
Process a single image by entering its filename.

#### 4. Exit
Quit the application.

## How It Works

### Face Detection
- Uses OpenCV's Haar Cascade classifier for face detection
- Converts images to grayscale for processing
- Returns bounding box coordinates for detected faces

### Face Encoding
- Creates simple encodings by resizing face regions to 100x100 pixels
- Flattens the image data to create a feature vector
- Falls back to full image encoding if no face is detected

### Face Recognition
- Compares test image encodings with known face encodings
- Uses Euclidean distance for similarity measurement
- Returns the best match if within tolerance threshold
- Calculates confidence score based on distance

### Results Display
- Shows bounding boxes around detected faces
- Labels known people with name and confidence percentage
- Marks unknown faces as "Unknown"
- Green boxes for known people, red for unknown

## Configuration

### Tolerance Adjustment
You can adjust the recognition tolerance in the `identifyPerson` function:
```python
def identifyPerson(test_image, tolerance=50000):
```
- Lower values = stricter matching
- Higher values = more lenient matching

### Supported Image Formats
The system supports these formats:
```python
supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.avif')
```

## Troubleshooting

### Common Issues

1. **"Folder not found" error**
   - Ensure `known_faces` and `test_images` folders exist
   - Check folder names are exactly as specified

2. **"No known faces found"**
   - Add images to the `known_faces` folder
   - Ensure images are in supported formats

3. **OpenCV display issues**
   - Run the display test (option 1) to check compatibility
   - Results will be saved to `output_results` folder if display fails

4. **Poor recognition accuracy**
   - Use clear, well-lit photos for known faces
   - Ensure faces are clearly visible and frontal
   - Adjust tolerance parameter if needed

### Performance Tips

- Use high-quality reference images in `known_faces`
- Ensure good lighting in test images
- One face per reference image works best
- Front-facing photos give better results

## Technical Details

### Dependencies Used
- **OpenCV**: Face detection, image processing, and display
- **NumPy**: Numerical operations and array handling
- **PIL (Pillow)**: Fallback image loading for unsupported formats

### Algorithm
1. Load and encode all known faces
2. For each test image:
   - Detect faces using Haar Cascade
   - Extract face region and create encoding
   - Compare with known encodings using Euclidean distance
   - Return best match within tolerance

### Limitations
- Simple encoding method (not deep learning based)
- Works best with frontal faces
- Limited to detecting one face per test image for identification
- Performance depends on image quality and lighting

## Future Enhancements

Potential improvements could include:
- Deep learning-based face recognition (dlib, face_recognition library)
- Multiple face detection and identification in single image
- Real-time video processing
- Database integration for larger datasets
- Web interface for easier usage

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

---

**Note**: This system is designed for educational and personal use. For production applications, consider using more advanced face recognition libraries like `face_recognition` or `dlib`.
