import sys
from pathlib import Path

# Add project root to the system path BEFORE importing from src
# app.py is in Yolo_11/, so parent is the project root
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import time
from PIL import Image, ImageDraw, ImageFont
from src.inference import YOLOv11Inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts

# streamlit run app.py
# Above code runs the application on port 8501

# streamlit run app.py --server.port 8080
# Above code runs the application on port 8080

def init_session_state():
    session_defaults = {
    "metadata" : None,
    "unique_classes" : [],
    "count_options" : {},
    "search_results" : [],
    "search_params" : {
        "search_mode" : "Any of selected classes (OR)",
        "selected_classes" : [],
        "thresholds" : {}
    } 
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes with labels on the image.
    
    Args:
        image: PIL Image object
        detections: List of detection dictionaries with 'bbox', 'class', 'confidence'
    
    Returns:
        PIL Image with drawn bounding boxes
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Color palette for different classes
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
    ]
    
    # Track colors assigned to each class
    class_colors = {}
    color_idx = 0
    
    for det in detections:
        bbox = det['bbox']
        cls = det['class']
        conf = det['confidence']
        
        # Assign color to class
        if cls not in class_colors:
            class_colors[cls] = colors[color_idx % len(colors)]
            color_idx += 1
        
        color = class_colors[cls]
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label text
        label = f"{cls} {conf:.2f}"
        
        # Get text bounding box for background
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Draw label background
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # Draw label text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
    
    return img_copy


init_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Computer Vision Powered Search Application")

# Main options
option = st.radio("Choose an option:",
                  ("Process new images", "Load existing metadata"),
                  horizontal=True)

if option == "Process new images":
    with st.expander("Process new images", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            image_dir = st.text_input("Image directory path:", placeholder="path/to/images")
        with col2:
            model_path = st.text_input("Model weights path:", "yolo11m.pt")

        if st.button("Start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer = YOLOv11Inference(model_path)
                        metadata = inferencer.process_directory(image_dir)
                        metadata_path = save_metadata(metadata, image_dir)
                        st.success(f"Processed {len(metadata)} images. Metadata saved to:")
                        st.code(str(metadata_path))
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
            else:
                st.warning(f"Please enter an image directory path")
else :
    with st.expander("Load Existing Metadata", expanded=True):
        metadata_path = st.text_input("Metadata file path:", placeholder="path/to/matadata.json")

        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    with st.spinner("Loading Metadata..."):
                        metadata = load_metadata(metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        st.success(f"Successfully loaded metadata for {len(metadata)} images.")
                except Exception as e:
                    st.error(f"Error loading metadata: {str(e)}")
            else:
                st.warning(f"Please enter a metadata file path")


                # Person, car, airplane, banana,apple
                # Person : 1,2,3,10

# st.write(f"{st.session_state.unique_classes}, {st.session_state.count_options}")

# Search Functionality
if st.session_state.metadata:
    st.header("Search Engine")

    # "search_params" : {
    #     "search_mode" : "Any of selected classes (OR)",
    #     "selected_classes" : [],
    #     "thresholds" : {}
    # } 

    with st.container():
        st.session_state.search_params["search_mode"] = st.radio("Search mode:", 
                ("Any of selected classes (OR)", "All selected classes (AND)"),
                horizontal=True
        )

        st.session_state.search_params["selected_classes"] = st.multiselect(
            "Classes to search for:", 
            options=st.session_state.unique_classes
        )

        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Thresholds (optional)")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max count for {cls}",
                        options=["None"] + st.session_state.count_options[cls]
                    )

        if st.button("Search Images", type="primary") and st.session_state.search_params["selected_classes"]:
            results = []
            search_params = st.session_state.search_params

            for item in st.session_state.metadata:
                matches = False
                class_matches = {}

                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item['detections'] if d['class'] == cls]
                    class_count = len(class_detections)
                    # 10 person
                    class_matches[cls] = False

                    threshold = search_params["thresholds"].get(cls, "None")
                    if threshold == "None":
                        class_matches[cls] = (class_count>=1)
                    else : 
                        class_matches[cls] = (class_count>=1 and class_count<= int(threshold))
                        # example 1: 
                        # threshold = 4
                        # class_count = 8
                        # then : class_matches[cls] = False
                        # We dont want to show this image

                        # example 2: 
                        # threshold = 4
                        # class_count = 2
                        # then : class_matches[cls] = True
                        # We want to show this image

                if search_params["search_mode"] == "Any of selected classes (OR)":
                    # not work only when both are not present or False
                    matches = any(class_matches.values())
                    # 1.jpg
                    # apple : False
                    # banana : True
                    # any(False, true) --> True
                else : # AND mode
                    # only work when both are present or True
                    matches = all(class_matches.values())
                    # 1.jpg
                    # apple : True
                    # banana : True
                    # any(False, true) --> True
                
                if matches:
                    results.append(item)

            st.session_state.search_results = results

        # Display search results with images
        if st.session_state.search_results:
            st.success(f"Found {len(st.session_state.search_results)} matching images")
            
            # Option to toggle bounding boxes
            show_boxes = st.checkbox("Show bounding boxes with labels", value=True)
            
            # Display images in a grid
            cols_per_row = 3
            for idx in range(0, len(st.session_state.search_results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for col_idx, col in enumerate(cols):
                    result_idx = idx + col_idx
                    if result_idx < len(st.session_state.search_results):
                        result = st.session_state.search_results[result_idx]
                        
                        with col:
                            try:
                                # Display image with bounding boxes
                                img_path = result['image_path']
                                img = Image.open(img_path)
                                
                                # Draw bounding boxes if enabled
                                if show_boxes:
                                    detections = result.get('detections', [])
                                    if detections:
                                        img_with_boxes = draw_bounding_boxes(img, detections)
                                        st.image(img_with_boxes, use_container_width=True)
                                    else:
                                        st.image(img, use_container_width=True)
                                else:
                                    st.image(img, use_container_width=True)
                                
                                # Display metadata
                                st.caption(f"**{Path(img_path).name}**")
                                
                                # Show detected objects
                                class_counts = result.get('class_counts', {})
                                if class_counts:
                                    st.write("Detected:")
                                    for cls, count in class_counts.items():
                                        st.write(f"• {cls}: {count}")
                                
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
        else:
            st.info("No images found matching the search criteria")


                    

        
        
