#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import os
import argparse
from typing import List, Tuple, Dict, Optional
import time
from pathlib import Path
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.yolo_model import YOLOv8n
from src.utils.logger import get_logger
from src.utils.config import config_manager
from src.utils.post_processing_simple import SimpleYOLOPostProcessor

logger = get_logger(__name__)

class YOLOv8nInferenceEngine:
    def __init__(self, model_path: str, config_path: str = "configs/config.yaml"):
        self.model_path = model_path
        self.config = config_manager.config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inference parameters
        self.confidence_threshold = 0.25
        self.nms_threshold = 0.45
        self.input_size = self.config['model']['input_size']
        self.num_classes = self.config['model']['num_classes']
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize post-processor
        self.post_processor = SimpleYOLOPostProcessor(
            num_classes=self.num_classes,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            input_size=self.input_size
        )
        
        # Class names
        self.class_names = ['room', 'window', 'door']
        
        # Colors for visualization
        self.colors = [
            (255, 0, 0),    # Red for rooms
            (0, 255, 0),    # Green for windows
            (0, 0, 255)     # Blue for doors
        ]
        
        logger.info(f"Inference engine initialized on {self.device}")
    
    def _load_model(self) -> YOLOv8n:
        logger.info(f"Loading model from {self.model_path}")
        
        model = YOLOv8n(
            num_classes=self.num_classes,
            input_size=self.input_size
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        
        logger.info("Model loaded successfully")
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_predictions(self, predictions: List[torch.Tensor], 
                               original_shape: Tuple[int, int]) -> List[Dict]:
        # Returns empty detections - to be reimplemented later
        return []
    
    def run_inference(self, image: np.ndarray) -> Tuple[List[Dict], float]:

        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        inference_time = time.time() - start_time
        
        # Post-process predictions using the post-processor
        original_shape = (image.shape[0], image.shape[1])
        detections = self.post_processor.process_predictions(predictions, original_shape)
        
        return detections, inference_time
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           inference_time: float) -> np.ndarray:
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # Get color for this class
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(vis_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label
            label = f"{self.class_names[class_id]}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (int(bbox[0]), int(bbox[1] - label_size[1] - 10)),
                         (int(bbox[0] + label_size[0]), int(bbox[1])),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, 
                       (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add inference time info
        time_text = f"Inference: {inference_time*1000:.1f}ms"
        cv2.putText(vis_image, time_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def batch_inference(self, images: List[np.ndarray]) -> List[Tuple[List[Dict], float]]:
        results = []
        
        for image in images:
            detections, inference_time = self.run_inference(image)
            results.append((detections, inference_time))
        
        return results
    
    def benchmark_inference(self, num_runs: int = 100) -> Dict[str, float]:

        logger.info(f"Benchmarking inference performance ({num_runs} runs)...")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _ = self.run_inference(dummy_image)
        
        # Benchmark
        times = []
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            _, inference_time = self.run_inference(dummy_image)
            times.append(inference_time)
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        metrics = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'fps': fps,
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
        
        logger.info(f"Average inference time: {avg_time*1000:.2f}ms")
        logger.info(f"FPS: {fps:.2f}")
        
        return metrics


def main(model_path: str = None, test_image: str = None, benchmark_runs: int = None):
    # Main inference function with config-first approach and CLI overrides
    try:
        # Load config
        from src.utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Apply CLI overrides
        if model_path is not None:
            config['inference']['model_path'] = model_path
            logger.info(f"CLI override: model_path = {model_path}")
        if test_image is not None:
            config['inference']['test_image'] = test_image
            logger.info(f"CLI override: test_image = {test_image}")
        if benchmark_runs is not None:
            config['inference']['benchmark_runs'] = benchmark_runs
            logger.info(f"CLI override: benchmark_runs = {benchmark_runs}")
        
        # Get paths from config
        model_path = config['inference']['model_path']
        test_image_path = config['inference']['test_image']
        benchmark_runs = config['inference']['benchmark_runs']
        
        # Check if trained model exists
        if not os.path.exists(model_path):
            logger.error(f"Trained model not found at {model_path}")
            logger.info("Please train the model first or specify --model-path")
            return
        
        # Create inference engine
        engine = YOLOv8nInferenceEngine(model_path)
        
        # Test on a sample image
        if os.path.exists(test_image_path):
            logger.info(f"Running inference on {test_image_path}")
            
            # Load image
            image = cv2.imread(test_image_path)
            if image is not None:
                # Run inference
                detections, inference_time = engine.run_inference(image)
                
                # Visualize results
                result_image = engine.visualize_detections(image, detections, inference_time)
                
                # Save result
                output_path = os.path.join(config['inference']['output_dir'], 'test_inference_result.jpg')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, result_image)
                
                logger.info(f"Inference completed in {inference_time*1000:.2f}ms")
                logger.info(f"Result saved to {output_path}")
            else:
                logger.error("Failed to load test image")
        else:
            logger.warning(f"Test image not found at {test_image_path}")
        
        # Run performance benchmark
        logger.info("Running performance benchmark...")
        benchmark_results = engine.benchmark_inference(num_runs=benchmark_runs)
        
        print("\n" + "="*50)
        print("INFERENCE BENCHMARK RESULTS")
        print("="*50)
        print(f"Average inference time: {benchmark_results['avg_inference_time']*1000:.2f}ms")
        print(f"FPS: {benchmark_results['fps']:.2f}")
        print(f"Min time: {benchmark_results['min_time']*1000:.2f}ms")
        print(f"Max time: {benchmark_results['max_time']*1000:.2f}ms")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    # Parse CLI arguments for overrides
    parser = argparse.ArgumentParser(description='Run YOLOv8n inference with config-first approach')
    parser.add_argument('--model-path', type=str, help='Override model path from config')
    parser.add_argument('--test-image', type=str, help='Override test image path from config')
    parser.add_argument('--benchmark-runs', type=int, help='Override number of benchmark runs from config')
    
    args = parser.parse_args()
    
    # Call main with CLI overrides
    main(
        model_path=args.model_path,
        test_image=args.test_image,
        benchmark_runs=args.benchmark_runs
    )
