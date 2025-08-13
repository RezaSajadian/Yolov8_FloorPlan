#!/usr/bin/env python3

import os
import json
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.config import ConfigManager
from src.utils.postprocessing_simple import SimpleYOLOPostProcessor

# Setup logging
logger = logging.getLogger(__name__)

class YOLOv8nModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()
        logger.info("Model loaded successfully")
        
    def _load_model(self):
        """Load the trained model."""
        try:
            # Load model architecture
            from src.models.yolo_model import YOLOv8n
            model = YOLOv8n(
                num_classes=self.config['model']['num_classes'],
                input_size=self.config['model']['input_size']
            )
            
            # Load checkpoint
            if self.model_path.endswith('.ckpt'):
                # PyTorch Lightning checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Direct model weights
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def benchmark_performance(self, num_runs: int = 100) -> Dict:
        # Benchmark model performance (FPS, latency)
        logger.info("Benchmarking model performance...")
        
        # Warmup
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark PyTorch
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        pytorch_time = (time.time() - start_time) / num_runs
        pytorch_fps = 1.0 / pytorch_time
        
        # Benchmark ONNX if available
        onnx_fps = 0
        onnx_time = 0
        try:
            onnx_path = self.convert_to_onnx()
            if onnx_path:
                import onnxruntime as ort
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                
                # Warmup
                dummy_input_np = dummy_input.cpu().numpy()
                for _ in range(10):
                    _ = session.run(None, {input_name: dummy_input_np})
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    _ = session.run(None, {input_name: dummy_input_np})
                onnx_time = (time.time() - start_time) / num_runs
                onnx_fps = 1.0 / onnx_time
                
        except Exception as e:
            logger.warning(f"ONNX benchmarking failed: {e}")
        
        results = {
            'pytorch': {
                'fps': pytorch_fps,
                'latency_ms': pytorch_time * 1000,
                'memory_mb': 0
            },
            'onnx': {
                'fps': onnx_fps,
                'latency_ms': onnx_time * 1000
            }
        }
        
        logger.info(f"PyTorch: {pytorch_fps:.2f} FPS, {pytorch_time*1000:.2f} ms")
        if onnx_fps > 0:
            logger.info(f"ONNX: {onnx_fps:.2f} FPS, {onnx_time*1000:.2f} ms")
        
        return results
    
    def convert_to_onnx(self) -> Optional[str]:
        try:
            logger.info("Converting model to ONNX format...")
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            
            # Export to ONNX
            onnx_path = "models/yolov8n.onnx"
            os.makedirs("models", exist_ok=True)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX model saved to {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return None
    
    def evaluate_on_test_data(self, test_data_path: str = None) -> Dict[str, float]:
        # Use config path if none provided
        if test_data_path is None:
            test_data_path = self.config['evaluation']['data_dir']
        
        logger.info(f"Evaluating model on test data: {test_data_path}")
        
        # Load test dataset
        from src.data.dataset import FloorplanDataset
        test_dataset = FloorplanDataset(
            data_dir=test_data_path,
            img_size=self.config['model']['input_size'],
            augment=False,
            is_training=False
        )
        
        if len(test_dataset) == 0:
            logger.error("No test data found")
            return {}
        
        # Limit evaluation to max_eval_images
        max_images = self.config['evaluation'].get('max_eval_images', len(test_dataset))
        eval_indices = list(range(min(max_images, len(test_dataset))))
        logger.info(f"Evaluating on {len(eval_indices)} images (max: {max_images})")
        
        # Evaluation metrics
        all_predictions = []
        all_ground_truth = []
        bbox_images = []  # Store images with bounding boxes for saving
        
        # Run inference on test data
        with torch.no_grad():
            for idx in tqdm(eval_indices, desc="Evaluating"):
                image, labels = test_dataset[idx]
                # For now, use the processed image since we don't have get_original_image method
                original_image = image  # We'll use the processed image for now
                image_tensor = image.unsqueeze(0).to(self.device)
                
                # Get predictions
                predictions = self.model(image_tensor)
                
                # Process predictions
                processed_preds = self._process_predictions(predictions)
                
                # Store for metrics calculation
                all_predictions.append(processed_preds)
                all_ground_truth.append(labels)
                
                # Store for bbox visualization
                bbox_images.append({
                    'image': original_image,
                    'predictions': processed_preds,
                    'ground_truth': labels,
                    'image_name': f"eval_{idx:03d}.jpg"
                })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_ground_truth)
        
        # Save images with bounding boxes if requested
        if self.config['evaluation'].get('save_bbox_images', False):
            self._save_bbox_images(bbox_images)
        
        logger.info(f"Evaluation completed. mAP: {metrics.get('mAP', 0):.4f}")
        return metrics
    
    def _process_predictions(self, predictions: List[torch.Tensor]) -> List:
        processed = []
        
        # Process predictions using the same post-processing as inference
        post_processor = SimpleYOLOPostProcessor(
            num_classes=self.config['model']['num_classes'],
            confidence_threshold=self.config['evaluation']['confidence_threshold'],
            nms_threshold=self.config['evaluation']['nms_threshold'],
            input_size=self.config['model']['input_size']
        )
        
        # Process each prediction
        for pred in predictions:
            # Use dummy shape for now - in real implementation, get actual image shape
            dummy_shape = (640, 640)
            detections = post_processor.process_predictions([pred], dummy_shape)
            processed.extend(detections)
        
        return processed
    
    def _calculate_metrics(self, predictions: List, ground_truth: List) -> Dict[str, float]:
        # Flatten predictions and ground truth
        all_preds = []
        all_gts = []
        
        for pred_list in predictions:
            if isinstance(pred_list, list):
                all_preds.extend(pred_list)
            else:
                all_preds.append(pred_list)
        
        for gt_list in ground_truth:
            if isinstance(gt_list, torch.Tensor):
                # Convert tensor to list format
                if gt_list.numel() > 0:
                    gt_data = gt_list.cpu().numpy()
                    for gt in gt_data:
                        if len(gt) >= 5:  # class_id, x_center, y_center, width, height
                            # Convert YOLO format to bbox format [x1, y1, x2, y2]
                            class_id, x_center, y_center, width, height = gt[:5]
                            x1 = x_center - width/2
                            y1 = y_center - height/2
                            x2 = x_center + width/2
                            y2 = y_center + height/2
                            all_gts.append({'bbox': [x1, y1, x2, y2], 'class_id': int(class_id)})
            elif isinstance(gt_list, list):
                all_gts.extend(gt_list)
        
        # Calculate IoU between predictions and ground truth
        if not all_preds or not all_gts:
            return {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        total_predictions = len(all_preds)
        total_ground_truth = len(all_gts)
        
        if total_predictions == 0:
            precision = 0.0
            recall = 0.0
        else:
            # Calculate IoU for each prediction-ground truth pair
            matched = 0
            for pred in all_preds:
                if isinstance(pred, dict) and 'bbox' in pred:
                    for gt in all_gts:
                        if isinstance(gt, dict) and 'bbox' in gt:
                            if self._calculate_iou(pred['bbox'], gt['bbox']) > 0.5:
                                matched += 1
                                break
            
            precision = matched / total_predictions if total_predictions > 0 else 0.0
            recall = matched / total_ground_truth if total_ground_truth > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'mAP': precision,  # Simplified mAP
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        # bbox format: [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _save_bbox_images(self, bbox_images: List[Dict]):
        import cv2
        import numpy as np
        
        output_dir = self.config['evaluation'].get('bbox_output_dir', 'evaluation_results/bbox_images')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving bounding box images to {output_dir}")
        
        for bbox_data in bbox_images:
            image = bbox_data['image']
            predictions = bbox_data['predictions']
            ground_truth = bbox_data['ground_truth']
            image_name = bbox_data['image_name']
            
            # Convert tensor to numpy array if needed
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                if image.shape[0] == 3:  # CHW format
                    image = np.transpose(image, (1, 2, 0))  # Convert to HWC
                image = (image * 255).astype(np.uint8)
            
            # Convert to BGR for OpenCV
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw predictions (red boxes)
            for pred in predictions:
                if isinstance(pred, dict) and 'bbox' in pred:
                    bbox = pred['bbox']
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = [int(coord * image.shape[1] if i < 2 else coord * image.shape[0]) for i, coord in enumerate(bbox)]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for predictions
                        
                        # Add class label if available
                        if 'class_id' in pred:
                            class_names = ['room', 'window', 'door']
                            class_name = class_names[pred['class_id']] if pred['class_id'] < len(class_names) else f"class_{pred['class_id']}"
                            cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw ground truth (green boxes)
            for gt in ground_truth:
                if isinstance(gt, torch.Tensor) and gt.numel() > 0:
                    gt_data = gt.cpu().numpy()
                    for gt_item in gt_data:
                        if hasattr(gt_item, '__len__') and len(gt_item) >= 5:  # class_id, x_center, y_center, width, height
                            class_id, x_center, y_center, width, height = gt_item[:5]
                            x1 = int((x_center - width/2) * image.shape[1])
                            y1 = int((y_center - height/2) * image.shape[0])
                            x2 = int((x_center + width/2) * image.shape[1])
                            y2 = int((y_center + height/2) * image.shape[0])
                            
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for ground truth
                            
                            # Add class label
                            class_names = ['room', 'window', 'door']
                            class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
                            cv2.putText(image, f"GT_{class_name}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the image
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, image)
        
        logger.info(f"Saved {len(bbox_images)} bounding box images")
    
    def run_comprehensive_evaluation(self, output_dir: str = "evaluation_results") -> Dict:
        """
        Run comprehensive evaluation including all metrics and comparisons.
        
        Args:
            output_dir: Directory to save evaluation results
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Performance benchmarking
        logger.info("1. Performance benchmarking...")
        performance_results = self.benchmark_performance()
        
        # 2. ONNX conversion
        logger.info("2. ONNX conversion...")
        onnx_path = self.convert_to_onnx()
        
        # 3. Test data evaluation
        logger.info("3. Test data evaluation...")
        evaluation_metrics = self.evaluate_on_test_data()
        
        # 4. Compile results
        results = {
            'performance': performance_results,
            'evaluation': evaluation_metrics,
            'model_info': {
                'model_path': self.model_path,
                'input_size': self.config['model']['input_size'],
                'num_classes': self.config['model']['num_classes'],
                'device': str(self.device)
            },
            'onnx_path': onnx_path
        }
        
        # 5. Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 6. Generate visualization
        self._generate_evaluation_plots(results, output_dir)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {output_dir}")
        return results
    
    def _generate_evaluation_plots(self, results: Dict, output_dir: str):
        """Generate evaluation plots and visualizations."""
        try:
            # Performance comparison plot
            if 'performance' in results and 'onnx' in results['performance']:
                import matplotlib.pyplot as plt
                
                # Performance comparison
                models = ['PyTorch', 'ONNX']
                fps_values = [results['performance']['pytorch']['fps'], results['performance']['onnx']['fps']]
                latency_values = [results['performance']['pytorch']['latency_ms'], results['performance']['onnx']['latency_ms']]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # FPS comparison
                bars1 = ax1.bar(models, fps_values, color=['#1f77b4', '#ff7f0e'])
                ax1.set_title('FPS Comparison')
                ax1.set_ylabel('Frames per Second')
                for bar, value in zip(bars1, fps_values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}', 
                            ha='center', va='bottom')
                
                # Latency comparison
                bars2 = ax2.bar(models, latency_values, color=['#1f77b4', '#ff7f0e'])
                ax2.set_title('Latency Comparison')
                ax2.set_ylabel('Latency (ms)')
                for bar, value in zip(bars2, latency_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}', 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Metrics summary
                if 'evaluation' in results:
                    metrics = results['evaluation']
                    metric_names = list(metrics.keys())
                    metric_values = list(metrics.values())
                    
                    plt.figure(figsize=(8, 6))
                    bars = plt.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
                    plt.title('Evaluation Metrics Summary')
                    plt.ylabel('Score')
                    plt.ylim(0, 1)
                    
                    for bar, value in zip(bars, metric_values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.4f}', 
                                ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8n model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, help='Path to test data directory')
    parser.add_argument('--batch-size', type=int, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = YOLOv8nModelEvaluator(args.model_path)
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nPerformance (PyTorch):")
        print(f"  FPS: {results['performance']['pytorch']['fps']:.2f}")
        print(f"  Latency: {results['performance']['pytorch']['latency_ms']:.2f} ms")
        
        if 'onnx' in results['performance'] and results['performance']['onnx']['fps'] > 0:
            print(f"\nPerformance (ONNX):")
            print(f"  FPS: {results['performance']['onnx']['fps']:.2f}")
            print(f"  Latency: {results['performance']['onnx']['latency_ms']:.2f} ms")
            onnx_improvement = ((results['performance']['onnx']['fps'] / results['performance']['pytorch']['fps']) - 1) * 100
            print(f"  FPS Improvement: {onnx_improvement:+.1f}%")
        
        print(f"\nEvaluation Metrics:")
        for metric, value in results['evaluation'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nResults saved to: evaluation_results/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
