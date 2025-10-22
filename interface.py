import streamlit as st
import torch
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from src.model.model import SiameseCNN
import torchvision.transforms as transforms

class SiameseCNNInterface:

    @st.cache_resource
    def load_model(_self, model_path):
        """Load and return the Siamese CNN model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = SiameseCNN(freeze_backbone=True)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, device

    @st.cache_data
    def load_test_data(_self, test_json_path):
        with open(test_json_path, 'r') as f:
            return json.load(f)

    def draw_bbox(_self, image, bbox, color=(0, 255, 0), label=""):
        """Draw a bounding box on the image."""
        img = image.copy()
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        if label:
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img

    def preprocess_image(self, image_path, bbox):
        """Load image, crop by bounding box, and preprocess for model."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        x, y, w, h = bbox
        crop = img[y:y+h, x:x+w]
        
        crop_pil = Image.fromarray(crop)
        tensor = transform(crop_pil).unsqueeze(0)
        
        return img, crop, tensor

    def main(_self):
        st.set_page_config(
            page_title="Siamese CNN Evaluator",
            layout="wide"
        )
        st.title("Siamese Convolutional Neural Network - Object pose evaluator")

        with st.sidebar:
            st.header("Settings")
            
            model_path = st.text_input("Model Path", value="checkpoints/best_checkpoint.pth")
            test_json_path = st.text_input("Test JSON Path", value="test_pairs.json")
            
            if st.button("Load Model and Data"):
                try:
                    st.session_state.model, st.session_state.device = _self.load_model(model_path)
                    st.session_state.test_pairs = _self.load_test_data(test_json_path)
                    st.success(f"Model and {len(st.session_state.test_pairs)} test pairs loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading model or data: {e}")
            
            st.markdown("---")
            st.markdown("Navigation")
        
        if 'model' not in st.session_state:
            st.warning("Please load the model and test data first.")
            return
        
        tab1, tab2 = st.tabs(["Single Pair Testing", "Batch Evaluation"])
        
        with tab1:
            st.header("Single Pair Testing")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                pair_idx = st.number_input(
                    "Pair Index",
                    min_value=0,
                    max_value=len(st.session_state.test_pairs) - 1,
                    value=0
                )
            
            with col2:
                if st.button("Previous"):
                    pair_idx = max(0, pair_idx - 1)
            
            with col3:
                if st.button("Next"):
                    pair_idx = min(len(st.session_state.test_pairs) - 1, pair_idx + 1)
            
            if st.button("Random Pair"):
                pair_idx = np.random.randint(0, len(st.session_state.test_pairs))
            
            pair = st.session_state.test_pairs[pair_idx]
            
            ref_full, ref_crop, ref_tensor = _self.preprocess_image(pair['reference_image'], pair['reference_bbox'])
            query_full, query_crop, query_tensor = _self.preprocess_image(pair['query_image'], pair['query_bbox'])

            ref_tensor = ref_tensor.to(st.session_state.device)
            query_tensor = query_tensor.to(st.session_state.device)
            
            with torch.no_grad():
                similarity, pred_angle = st.session_state.model(ref_tensor, query_tensor)
            
            pred_angle = pred_angle.item() * 180.0
            similarity = similarity.item()
            
            angle_error = abs(pred_angle - pair['angle_difference'])
            pred_match = 1 if similarity > 0.5 else 0
            match_correct = (pred_match == pair['match_label'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Reference Object")
                ref_viz = _self.draw_bbox(ref_full, pair['reference_bbox'], (0, 255, 0), "Reference")
                st.image(ref_viz, use_container_width=True)
                st.image(ref_crop, caption="Cropped", use_container_width=True)
            with col2:
                st.subheader("Query Object")
                query_viz = _self.draw_bbox(query_full, pair['query_bbox'], (255, 0, 0), "Query")
                st.image(query_viz, use_container_width=True)
                st.image(query_crop, caption="Cropped", use_container_width=True)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Similarity Score", f"{similarity:.3f}", delta=f"{(similarity - 0.5)*100:.1f}% from threshold")
            with col2:
                st.metric("Predicted Angle", f"{pred_angle:.2f}°", delta=f"Error: {angle_error:.2f}°")
            with col3:
                st.metric("Match Prediction", "MATCH" if pred_match == 1 else "NO MATCH",
                        delta="Correct" if match_correct else "Incorrect")
            
            with st.expander("Detailed Results"):
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
                    st.success(f"Angle error: {angle_error:.2f}° (Good)")
                elif angle_error < 20:
                    st.warning(f"Angle error: {angle_error:.2f}° (Fair)")
                else:
                    st.error(f"Angle error: {angle_error:.2f}° (Poor)")
                
                if match_correct:
                    st.success("Match prediction is correct.")
                else:
                    st.error("Match prediction is incorrect.")
        
        with tab2:
            st.header("Batch Evaluation")
            
            batch_size = st.slider("Number of samples to test", min_value=10,
                                max_value=min(500, len(st.session_state.test_pairs)), value=100, step=10)
            
            if st.button("Run Batch Test"):
                with st.spinner("Evaluating..."):
                    results = {'angle_errors': [], 'match_correct': [], 'similarities': []}
                    progress_bar = st.progress(0)
                    
                    for i in range(batch_size):
                        pair = st.session_state.test_pairs[i]
                        _, _, ref_tensor = _self.preprocess_image(pair['reference_image'], pair['reference_bbox'])
                        _, _, query_tensor = _self.preprocess_image(pair['query_image'], pair['query_bbox'])

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
                    
                    accuracy = np.mean(results['match_correct']) * 100
                    mae = np.mean(results['angle_errors'])
                    rmse = np.sqrt(np.mean(np.array(results['angle_errors'])**2))
                    within_10deg = np.mean([e < 10 for e in results['angle_errors']]) * 100
                    
                    st.success("Batch evaluation complete.")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2f}%")
                    col2.metric("MAE", f"{mae:.2f}°")
                    col3.metric("RMSE", f"{rmse:.2f}°")
                    col4.metric("Within ±10°", f"{within_10deg:.1f}%")
                    
                    st.subheader("Error Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(results['angle_errors'], bins=30, edgecolor='black', alpha=0.7)
                    ax.axvline(mae, color='red', linestyle='--', label=f'MAE: {mae:.2f}°')
                    ax.set_xlabel('Angle Error (degrees)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Angle Prediction Errors')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

if __name__ == "__main__":
    interface = SiameseCNNInterface()
    interface.main()