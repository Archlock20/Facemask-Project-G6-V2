from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
from deeplearning import face_mask_prediction

class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)
    change_label_signal = qtc.pyqtSignal(str) # señal para actualizar el contador de mascarillas
    
    def __init__(self, mask_label):
        super().__init__()
        self.run_flag = True
        self.mask_label = mask_label
        
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while self.run_flag:
            ret , frame = cap.read()
            prediction_img = face_mask_prediction(frame)
            mask_count = 0
            
            if ret == True:
                self.change_pixmap_signal.emit(prediction_img[0])
                # emitir señal para actualizar el contador de mascarillas
                self.change_label_signal.emit("Personas con mascarillas: {}".format(mask_count))
                
        cap.release()
        
    def stop(self):
        self.run_flag =False
        self.wait()
        

class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('./images/icon.png'))
        self.setWindowTitle('Software de deteccion de mascarillas')
        self.setFixedSize(600,600)
        
        # Añadir Widgets
        label = qtw.QLabel('<h2>Aplicacion de reconocimiento de mascarilla facial</h2>')
        self.cameraButton = qtw.QPushButton('Activar camara', clicked=self.cameraButtonClick, checkable=True)
        
        # Pantalla
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(600,480)
        self.img.fill(qtg.QColor('darkGrey'))
        self.screen.setPixmap(self.img)
        self.mask_label = qtw.QLabel(self)
        
        # Layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.screen)
        
        self.setLayout(layout)
        self.show()
    
    def cameraButtonClick(self):
        print('cliqueado')
        status = self.cameraButton.isChecked()
        if status == True:
            self.cameraButton.setText('Cerrar camara')
            
            # Abrir la camara
            #self.capture = VideoCapture()
            self.capture = VideoCapture(self.mask_label)
            self.capture.change_pixmap_signal.connect(self.updateImage)
            self.capture.start()
            
        elif status == False:
            self.cameraButton.setText('Abrir camara')
            self.capture.stop()
            
    @qtc.pyqtSlot(np.ndarray)
    def updateImage(self,image_array):
        rgb_img = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_lines = ch*w
        # Convertir en QImage
        convertedImage = qtg.QImage(rgb_img.data,w,h,bytes_per_lines,qtg.QImage.Format_RGB888)
        scaledImage = convertedImage.scaled(600,480,qtc.Qt.KeepAspectRatio)
        qt_img = qtg.QPixmap.fromImage(scaledImage)
        
        # actualizar la pantalla
        self.screen.setPixmap(qt_img) 


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec())