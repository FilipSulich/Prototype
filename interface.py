# gui_streamlit.py

import streamlit as st
import torch
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import your model
from src.model.model import SiameseCNN
import torchvision.transforms as transforms

st.set_page_config(
    page_title="Siamese CNN Evaluator",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load model (cached)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SiameseCNN(pretrained=False, freeze_backbone=True)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device

@st.cache_data
def load_test_data(test_json_path):
    """Load test data (cached)"""
    with open(test_json_path, 'r') as f:
        return json.load(f)

def draw_bbox(image, bbox, color=(0, 255, 0), label=""):
    """Draw bounding box"""
    img = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
    
    if label:
        cv2.putText(img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img

def preprocess_image(image_path, bbox):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x, y, w, h = bbox
    crop = img[y:y+h, x:x+w]
    
    crop_pil = Image.fromarray(crop)
    tensor = transform(crop_pil).unsqueeze(0)
    
    return img, crop, tensor

def main():
    st.title("🤖 Siamese CNN - Pose Comparison Evaluator")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_path = st.text_input(
            "Model Path",
            value="checkpoints/best_checkpoint.pth"
        )
        
        test_json_path = st.text_input(
            "Test JSON Path",
            value="test_pairs.json"
        )
        
        if st.button("🔄 Load Model & Data"):
            try:
                st.session_state.model, st.session_state.device = load_model(model_path)
                st.session_state.test_pairs = load_test_data(test_json_path)
                st.success(f"✓ Loaded model and {len(st.session_state.test_pairs)} test pairs")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
    
    # Check if model loaded
    if 'model' not in st.session_state:
        st.warning("👆 Please load model and data from sidebar")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Single Pair Testing", "📊 Batch Evaluation"])
    
    with tab1:
        st.header("Single Pair Testing")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pair_idx = st.number_input(
                "Pair Index",
                min_value=0,
                max_value=len(st.session_state.test_pairs)-1,
                value=0
            )
        
        with col2:
            if st.button("⬅ Previous"):
                pair_idx = max(0, pair_idx - 1)
        
        with col3:
            if st.button("Next ➡"):
                pair_idx = min(len(st.session_state.test_pairs) - 1, pair_idx + 1)
        
        if st.button("🎲 Random Pair"):
            pair_idx = np.random.randint(0, len(st.session_state.test_pairs))
        
        # Load pair
        pair = st.session_state.test_pairs[pair_idx]
        
        # Preprocess
        ref_full, ref_crop, ref_tensor = preprocess_image(
            pair['reference_image'],
            pair['reference_bbox']
        )
        query_full, query_crop, query_tensor = preprocess_image(
            pair['query_image'],
            pair['query_bbox']
        )
        
        # Predict
        ref_tensor = ref_tensor.to(st.session_state.device)
        query_tensor = query_tensor.to(st.session_state.device)
        
        with torch.no_grad():
            similarity, pred_angle = st.session_state.model(ref_tensor, query_tensor)
        
        pred_angle = pred_angle.item() * 180.0
        similarity = similarity.item()
        
        # Calculate errors
        angle_error = abs(pred_angle - pair['angle_difference'])
        pred_match = 1 if similarity > 0.5 else 0
        match_correct = (pred_match == pair['match_label'])
        
        # Display images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Reference Object")
            ref_viz = draw_bbox(ref_full, pair['reference_bbox'], (0, 255, 0), "Reference")
            st.image(ref_viz, use_column_width=True)
            st.image(ref_crop, caption="Cropped", use_column_width=True)
        
        with col2:
            st.subheader("📸 Query Object")
            query_viz = draw_bbox(query_full, pair['query_bbox'], (255, 0, 0), "Query")
            st.image(query_viz, use_column_width=True)
            st.image(query_crop, caption="Cropped", use_column_width=True)
        
        # Results
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Similarity Score",
                f"{similarity:.3f}",
                delta=f"{(similarity - 0.5)*100:.1f}% from threshold"
            )
        
        with col2:
            st.metric(
                "Predicted Angle",
                f"{pred_angle:.2f}°",
                delta=f"Error: {angle_error:.2f}°"
            )
        
        with col3:
            st.metric(
                "Match Prediction",
                "✓ MATCH" if pred_match == 1 else "✗ NO MATCH",
                delta="Correct" if match_correct else "Wrong"
            )
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ground Truth:**")
                st.write(f"- True Angle: {pair['angle_difference']:.2f}°")
                st.write(f"- True Match: {pair['match_label']}")
                st.write(f"- Object ID: {pair['object_id']}")
            
            with col2:
                st.markdown("**Predictions:**")
                st.write(f"- Predicted Angle: {pred_angle:.2f}°")
                st.write(f"- Similarity: {similarity:.3f}")
                st.write(f"- Predicted Match: {pred_match}")
            
            st.markdown("**Error Analysis:**")
            if angle_error < 10:
                st.success(f"✓ Angle error: {angle_error:.2f}° (GOOD)")
            elif angle_error < 20:
                st.warning(f"⚠ Angle error: {angle_error:.2f}° (FAIR)")
            else:
                st.error(f"✗ Angle error: {angle_error:.2f}° (POOR)")
            
            if match_correct:
                st.success("✓ Match prediction: CORRECT")
            else:
                st.error("✗ Match prediction: WRONG")
    
    with tab2:
        st.header("Batch Evaluation")
        
        batch_size = st.slider(
            "Number of samples to test",
            min_value=10,
            max_value=min(500, len(st.session_state.test_pairs)),
            value=100,
            step=10
        )
        
        if st.button("▶ Run Batch Test"):
            with st.spinner("Testing..."):
                results = {
                    'angle_errors': [],
                    'match_correct': [],
                    'similarities': []
                }
                
                progress_bar = st.progress(0)
                
                for i in range(batch_size):
                    pair = st.session_state.test_pairs[i]
                    
                    _, _, ref_tensor = preprocess_image(
                        pair['reference_image'],
                        pair['reference_bbox']
                    )
                    _, _, query_tensor = preprocess_image(
                        pair['query_image'],
                        pair['query_bbox']
                    )
                    
                    ref_tensor = ref_tensor.to(st.session_state.device)
                    query_tensor = query_tensor.to(st.session_state.device)
                    
                    with torch.no_grad():
                        similarity, pred_angle = st.session_state.model(ref_tensor, query_tensor)
                    
                    pred_angle = pred_angle.item() * 180.0
                    similarity = similarity.item()
                    
                    angle_error = abs(pred_angle - pair['angle_difference'])
                    pred_match = 1 if similarity > 0.5 else 0
                    match_correct = (pred_match == pair['match_label'])
                    
                    results['angle_errors'].append(angle_error)
                    results['match_correct'].append(match_correct)
                    results['similarities'].append(similarity)
                    
                    progress_bar.progress((i + 1) / batch_size)
                
                # Calculate metrics
                accuracy = np.mean(results['match_correct']) * 100
                mae = np.mean(results['angle_errors'])
                rmse = np.sqrt(np.mean(np.array(results['angle_errors'])**2))
                within_10deg = np.mean([e < 10 for e in results['angle_errors']]) * 100
                within_20deg = np.mean([e < 20 for e in results['angle_errors']]) * 100
                
                # Display metrics
                st.success("✓ Batch test complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                
                with col2:
                    st.metric("MAE", f"{mae:.2f}°")
                
                with col3:
                    st.metric("RMSE", f"{rmse:.2f}°")
                
                with col4:
                    st.metric("Within ±10°", f"{within_10deg:.1f}%")
                
                # Histogram
                st.subheader("📊 Error Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(results['angle_errors'], bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(mae, color='red', linestyle='--', label=f'MAE: {mae:.2f}°')
                ax.set_xlabel('Angle Error (degrees)')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Angle Prediction Errors')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

if __name__ == "__main__":
    main()