import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit, 
                             QGroupBox, QFormLayout, QLineEdit, QDoubleSpinBox, QCheckBox,
                             QMessageBox, QTabWidget, QListWidget, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from inference import MMR_Detector
from config import get_cfg


class InferenceThread(QThread):
    """Thread for running inference to prevent UI blocking"""
    progress_update = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(str, str, float, str)  # image_path, decision, score, output_path
    batch_complete = pyqtSignal(list)  # list of results
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, detector, image_paths, threshold):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths
        self.threshold = threshold

    def run(self):
        try:
            if len(self.image_paths) == 1:
                # Single image detection
                image_path = self.image_paths[0]
                decision, score, anomaly_map = self.detector.detect(image_path, self.threshold)
                # Get output path (same logic as in inference.py)
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                # Determine output path based on decision
                if "OK" in decision:
                    output_path = os.path.join(self.detector.ok_dir, f"{name}_detection_result_high_res.png")
                else:  # NG
                    output_path = os.path.join(self.detector.ng_dir, f"{name}_detection_result_high_res.png")
                self.result_ready.emit(image_path, decision, score, output_path)
            else:
                # Batch detection
                results = self.detector.batch_detect(self.image_paths, self.threshold)
                self.batch_complete.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))


class MMRAnomalyDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.current_image_path = None
        self.output_image_path = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('MMR工业异常检测系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Single image tab
        single_tab = self.create_single_tab()
        tab_widget.addTab(single_tab, "单张图像检测")
        
        # Batch tab
        batch_tab = self.create_batch_tab()
        tab_widget.addTab(batch_tab, "批量图像检测")
        
        # Configuration tab
        config_tab = self.create_config_tab()
        tab_widget.addTab(config_tab, "配置与日志")
        
        # Create status bar
        self.status_bar = self.statusBar()
        
    def create_single_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left panel for controls
        left_panel = QGroupBox("控制面板")
        left_layout = QVBoxLayout(left_panel)
        
        # Model selection
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout(model_group)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText("D:\Project\MMR-master\log_MMR_AeBAD_S_54\checkpoints\mmr_model_final.pth")
        model_browse_btn = QPushButton("浏览...")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addRow("模型路径:", self.model_path_edit)
        model_layout.addRow("", model_browse_btn)
        
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setText("D:\Project\MMR-master\method_config/AeBAD_S/MMR.yaml")
        config_browse_btn = QPushButton("浏览...")
        config_browse_btn.clicked.connect(self.browse_config)
        model_layout.addRow("配置文件:", self.config_path_edit)
        model_layout.addRow("", config_browse_btn)
        
        # Output directory
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText("detection_results")
        output_dir_browse_btn = QPushButton("浏览...")
        output_dir_browse_btn.clicked.connect(self.browse_output_dir)
        model_layout.addRow("输出目录:", self.output_dir_edit)
        model_layout.addRow("", output_dir_browse_btn)
        
        # Threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.6)
        model_layout.addRow("阈值:", self.threshold_spin)
        
        # Load model button
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addRow("", self.load_model_btn)
        
        left_layout.addWidget(model_group)
        
        # Image selection
        image_group = QGroupBox("图像选择")
        image_layout = QVBoxLayout(image_group)
        
        self.image_path_edit = QLineEdit()
        image_browse_btn = QPushButton("浏览图像...")
        image_browse_btn.clicked.connect(self.browse_image)
        image_layout.addWidget(QLabel("图像路径:"))
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(image_browse_btn)
        
        self.detect_btn = QPushButton("检测异常")
        self.detect_btn.clicked.connect(self.detect_single)
        self.detect_btn.setEnabled(False)  # Enable after model is loaded
        image_layout.addWidget(self.detect_btn)
        
        left_layout.addWidget(image_group)
        
        # Results
        results_group = QGroupBox("检测结果")
        results_layout = QVBoxLayout(results_group)
        
        self.result_label = QLabel("检测结果: 无")
        self.score_label = QLabel("异常分数: 无")
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.score_label)
        
        left_layout.addWidget(results_group)
        left_layout.addStretch()
        
        # Right panel for image display
        right_panel = QGroupBox("图像显示")
        right_layout = QVBoxLayout(right_panel)
        
        self.image_label = QLabel("未加载图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        right_layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)
        
        return tab
        
    def create_batch_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left panel for controls
        left_panel = QGroupBox("批量处理")
        left_layout = QVBoxLayout(left_panel)
        
        # Folder selection
        folder_group = QGroupBox("文件夹选择")
        folder_layout = QVBoxLayout(folder_group)
        
        self.folder_path_edit = QLineEdit()
        folder_browse_btn = QPushButton("浏览文件夹...")
        folder_browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(QLabel("文件夹路径:"))
        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(folder_browse_btn)
        
        self.batch_detect_btn = QPushButton("检测所有图像")
        self.batch_detect_btn.clicked.connect(self.detect_batch)
        self.batch_detect_btn.setEnabled(False)  # Enable after model is loaded
        folder_layout.addWidget(self.batch_detect_btn)
        
        left_layout.addWidget(folder_group)
        
        # Results list
        results_group = QGroupBox("检测结果列表")
        results_layout = QVBoxLayout(results_group)
        
        self.results_list = QListWidget()
        results_layout.addWidget(self.results_list)
        
        left_layout.addWidget(results_group)
        left_layout.addStretch()
        
        # Right panel for image display
        right_panel = QGroupBox("选中结果")
        right_layout = QVBoxLayout(right_panel)
        
        self.batch_image_label = QLabel("未选择结果")
        self.batch_image_label.setAlignment(Qt.AlignCenter)
        self.batch_image_label.setMinimumSize(400, 400)
        right_layout.addWidget(self.batch_image_label)
        
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)
        
        # Connect list selection to image display
        self.results_list.itemClicked.connect(self.display_batch_result)
        
        return tab
        
    def create_config_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Log display
        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        layout.addWidget(log_group)
        
        return tab
        
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pth *.pt)")
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "YAML文件 (*.yaml *.yml)")
        if file_path:
            self.config_path_edit.setText(file_path)
            
    def browse_output_dir(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder_path:
            self.output_dir_edit.setText(folder_path)
            
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            self.image_path_edit.setText(file_path)
            self.display_image(file_path)
            
    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            
    def load_model(self):
        model_path = self.model_path_edit.text()
        config_path = self.config_path_edit.text()
        output_dir = self.output_dir_edit.text()
        
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "错误", f"模型文件未找到: {model_path}")
            return
            
        if not os.path.exists(config_path):
            QMessageBox.critical(self, "错误", f"配置文件未找到: {config_path}")
            return
            
        try:
            self.status_bar.showMessage("正在加载模型...")
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            
            self.detector = MMR_Detector(cfg, model_path, output_dir)
            self.detect_btn.setEnabled(True)
            self.batch_detect_btn.setEnabled(True)
            self.status_bar.showMessage("模型加载成功!", 3000)
            self.log_display.append("模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.status_bar.showMessage("模型加载失败", 3000)
            self.log_display.append(f"模型加载错误: {str(e)}")
            
    def detect_single(self):
        if not self.detector:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        image_path = self.image_path_edit.text()
        if not os.path.exists(image_path):
            QMessageBox.critical(self, "错误", f"图像文件未找到: {image_path}")
            return
            
        threshold = self.threshold_spin.value()
        
        # Start inference in separate thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.detect_btn.setEnabled(False)
        
        self.inference_thread = InferenceThread(self.detector, [image_path], threshold)
        self.inference_thread.result_ready.connect(self.on_single_result)
        self.inference_thread.error_occurred.connect(self.on_error)
        self.inference_thread.start()
        
    def on_single_result(self, image_path, decision, score, output_path):
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        
        # Update results
        self.result_label.setText(f"检测结果: {decision}")
        self.score_label.setText(f"异常分数: {score:.4f}")
        
        # Display output image if it exists
        if os.path.exists(output_path):
            self.display_image(output_path)
            self.output_image_path = output_path
        else:
            self.image_label.setText("输出图像未找到")
            
        self.status_bar.showMessage(f"检测完成: {decision}", 3000)
        self.log_display.append(f"单张图像检测: {os.path.basename(image_path)} - {decision} (分数: {score:.4f})")
        
    def detect_batch(self):
        if not self.detector:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        folder_path = self.folder_path_edit.text()
        if not os.path.exists(folder_path):
            QMessageBox.critical(self, "错误", f"文件夹未找到: {folder_path}")
            return
            
        # Get all image files in folder
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for extension in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, extension)))
            image_paths.extend(glob.glob(os.path.join(folder_path, extension.upper())))
            
        if not image_paths:
            QMessageBox.warning(self, "警告", "在所选文件夹中未找到图像文件")
            return
            
        threshold = self.threshold_spin.value()
        
        # Start batch inference in separate thread
        self.batch_detect_btn.setEnabled(False)
        self.results_list.clear()
        self.batch_image_label.setText("处理中...")
        
        self.inference_thread = InferenceThread(self.detector, image_paths, threshold)
        self.inference_thread.batch_complete.connect(self.on_batch_complete)
        self.inference_thread.error_occurred.connect(self.on_error)
        self.inference_thread.start()
        
    def on_batch_complete(self, results):
        self.batch_detect_btn.setEnabled(True)
        self.results_list.clear()
        
        for image_path, decision, score, anomaly_map in results:
            if decision != "ERROR":
                item_text = f"{os.path.basename(image_path)} - {decision} ({score:.4f})"
                self.results_list.addItem(item_text)
            else:
                item_text = f"{os.path.basename(image_path)} - 错误"
                self.results_list.addItem(item_text)
                
        self.status_bar.showMessage(f"批量检测完成. 处理了 {len(results)} 张图像.", 5000)
        self.log_display.append(f"批量检测完成. 处理了 {len(results)} 张图像.")
        # Update the batch image label to show that processing is complete
        if self.results_list.count() > 0:
            self.batch_image_label.setText("请选择结果查看")
        else:
            self.batch_image_label.setText("未找到结果")
        
    def on_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.detect_btn.setEnabled(True)
        self.batch_detect_btn.setEnabled(True)
        self.batch_image_label.setText("未选择结果")
        QMessageBox.critical(self, "错误", f"检测失败: {error_msg}")
        self.status_bar.showMessage("检测失败", 3000)
        self.log_display.append(f"错误: {error_msg}")
        
    def display_image(self, image_path):
        # Load image using OpenCV for better compatibility
        image = cv2.imread(image_path)
        if image is None:
            self.image_label.setText("图像加载失败")
            return
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
    def display_batch_result(self, item):
        # Extract image name from item text
        item_text = item.text()
        image_name = item_text.split(" - ")[0]
        
        # Check if detector is loaded
        if self.detector is None:
            self.batch_image_label.setText("请先加载模型")
            return
        
        # Find the output image in either OK or NG directory
        ok_image_path = os.path.join(self.detector.ok_dir, f"{image_name.split('.')[0]}_detection_result_high_res.png")
        ng_image_path = os.path.join(self.detector.ng_dir, f"{image_name.split('.')[0]}_detection_result_high_res.png")
        
        if os.path.exists(ok_image_path):
            self.display_image(ok_image_path)
        elif os.path.exists(ng_image_path):
            self.display_image(ng_image_path)
        else:
            # Try to display original image
            folder_path = self.folder_path_edit.text()
            original_image_path = os.path.join(folder_path, image_name)
            if os.path.exists(original_image_path):
                self.display_image(original_image_path)
            else:
                self.batch_image_label.setText("图像未找到")


def main():
    app = QApplication(sys.argv)
    gui = MMRAnomalyDetectionGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()