import os
import sys
import shutil
import os
import patoolib
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QScrollArea, QGroupBox, QFormLayout,
                             QPushButton, QCheckBox, QFileDialog, QVBoxLayout, QProgressBar, QHBoxLayout)
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf


class nn_error(QWidget):  # _________________________________________ Класс окна при остутствии модели в директории
    def __init__(self):
        super().__init__()
        self.setGeometry(800, 400, 175, 50)
        self.label = QLabel(self)
        self.label.setText("Модель не обнаружена")
        self.label.move(10, 10)


class context_window(QWidget):
    def __init__(self, arr):
        # Надо не забыть вызвать инициализатор базового класса
        super().__init__()
        self.widget = QWidget(self)
        self.predict_arr = arr
        self.setStyleSheet("background-color: rgb(121, 217, 217)")
        self.scrollbar = QScrollArea(self)
        self.scrollbar.resize(100, 1000)
        self.scrollbar.move(1400, 100)
        self.results_img = QVBoxLayout(self)
        self.init_ui()

    def init_ui(self):
        self.setGeometry(50, 50, 1200, 800)
        dir_list = os.listdir(__file__ + "/../Interface_entry/No_Sorted/")
        test_label = QLabel(self.widget)
        form_layout = QFormLayout(test_label)
        form_layout.addRow(test_label, test_label)
        group_box = QGroupBox("Предсказание нейросети")
        for i in dir_list:
            j = dir_list.index(i)
            pixmap = QPixmap(__file__ + "/../Interface_entry/No_Sorted/" + i)
            pixmap = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
            label_image = QLabel(self)
            label_image.resize(300, 300)
            label_image.setPixmap(pixmap)
            bar = QProgressBar(self)
            status, percent = self.predict(self.predict_arr[j])
            percent_normal = QLabel(self)
            percent_pneumonia = QLabel(self)
            lable_lis = QLabel(self)
            lable_lis.setText(i)
            if status == 1:
                percent_normal.setText("    " + str(format(percent, 'f')) + "%")
                percent_pneumonia.setText(str(format((100 - percent), 'f')) + "%")
                bar.setValue(percent)
            else:
                percent_pneumonia.setText("    " + str(format(percent, 'f')) + "%")
                percent_normal.setText(str(format((100 - percent), 'f')) + "%")
                bar.setValue(100 - percent)
            bar.setStyleSheet("QProgressBar{"
                              "background-color: #ff0000;"
                              "border: 2px solid grey;"
                              "border-radius: 5px;"
                              "color: #00ff00;"
                              "}"
                              "QProgressBar::chunk"
                              "{background-color: #00ff00;"
                              "border-radius: 15px;"
                              "}")
            layout_horizontal = QHBoxLayout(self)
            layout_horizontal.addWidget(lable_lis)
            layout_horizontal.addWidget(percent_normal)
            layout_horizontal.addWidget(bar)
            layout_horizontal.addWidget(percent_pneumonia)
            form_layout.addRow(label_image, layout_horizontal)
        group_box.setLayout(form_layout)
        self.scrollbar.setWidget(group_box)
        self.scrollbar.setWidgetResizable(True)
        self.scrollbar.setFixedHeight(800)
        self.results_img.addWidget(self.scrollbar)

    def predict(self, arr):
        distance1 = abs(float(arr[0]) - 1)
        distance2 = abs(float(arr[1]) - 1)
        if distance1 > distance2:
            return 1, (distance1 / (distance2 + distance1) * 100)
        return 2, (distance2 / (distance2 + distance1) * 100)

    def closeEvent(self, event):  # ____________________________ Удаление системных файлов при закрытии окна
        img_list_entry = os.listdir("Interface_entry/No_Sorted/")
        for i in img_list_entry:
            os.remove("Interface_entry/No_Sorted/" + i)
        img_list_exit_normal = os.listdir("Interface_exit/Normal")
        # print(img_list_exit_normal)
        img_list_exit_pneumonia = os.listdir("Interface_exit/Pneumonia")
        # print(img_list_exit_pneumonia)
        for j in img_list_exit_normal:
            os.remove("Interface_exit/Normal/" + j)
        for k in img_list_exit_pneumonia:
            os.remove("Interface_exit/Pneumonia/" + k)


class main_Interface(QWidget):
    def __init__(self):
        # Надо не забыть вызвать инициализатор базового класса
        super().__init__()
        # В метод initUI() будем выносить всю настройку интерфейса,
        # чтобы не перегружать инициализатор
        self.nn_accurate = QLabel(self)
        self.nn_info = QLabel(self)
        self.setStyleSheet("background-color: rgb(121, 217, 217)")
        self.buttonArch = QPushButton("Выберите .rar архив", self)
        self.checkboxArch = QCheckBox("Удалить файл?", self)
        self.buttonImage = QPushButton("Выберите изображение", self)
        self.checkboxImage = QCheckBox("Удалить файл?", self)
        self.labelXIPA = QLabel(self)
        self.label_about = QLabel(self)
        self.prepare_data = QLabel(self)
        self.new_window = None
        self.nn_class = NeuralNetwork("Interface_entry", "Interface_exit")
        self.init_ui()

    def init_ui(self):
        # Зададим размер и положение нашего виджета,
        self.setGeometry(850, 100, 900, 700)
        # А также его заголовок
        self.setWindowTitle('XIP Application')
        self.labelXIPA.setText("X-ray Identification Pneumonia Application")
        self.labelXIPA.move(60, 20)
        self.labelXIPA.setFont(QtGui.QFont('SansSerif', 24, QtGui.QFont.Bold))
        self.labelXIPA.adjustSize()
        self.label_about.setWordWrap(True)
        self.label_about.resize(800, 50)
        self.label_about.setText("Добро пожаловать в приложение XIPA. Это приложение облегчит мед персоналу "
                                 "прогнозирование пневмонии. Основой приложения выступает нейросеть, "
                                 "обученная на 6000 тыс фотографий здоровых и больных пневмонией. "
                                 "Данное приложениене несёт лишь ознакомительный "
                                 "характер. Требуется рекомендация со специалистом")
        self.label_about.move(60, 70)
        self.label_about.setFont(QtGui.QFont('SansSerif', 8, QtGui.QFont.ExtraLight))
        self.prepare_data.setWordWrap(True)
        self.prepare_data.resize(600, 70)
        self.prepare_data.setText("XIPA может обработать rar файлы и изображения отдельно, выберите нужное ниже:")
        self.prepare_data.move(60, 140)
        self.prepare_data.setFont(QtGui.QFont('SansSerif', 15))

        self.buttonArch.resize(150, 50)
        self.buttonArch.move(50, 250)
        self.buttonArch.setStyleSheet("QPushButton{background-color: rgb(18, 199, 51);"
                                      "border-style: outset;"
                                      "border-width: 2px;"
                                      "border-radius: 10px;"
                                      "border-color: rgb(0, 130, 24);"
                                      "font: bold 14px;"
                                      "min-width: 10em;"
                                      "padding: 6px;"
                                      "}"
                                      "QPushButton:pressed{"
                                      "background-color: rgb(0, 130, 24);"
                                      "border-style: inset;"
                                      "}"
                                      )
        self.checkboxArch.move(50, 300)
        self.buttonArch.clicked.connect(self.button_archive_push)
        self.buttonImage.resize(200, 50)
        self.buttonImage.move(600, 250)
        self.buttonImage.setStyleSheet("QPushButton{background-color: rgb(245, 217, 0);"
                                       "border-style: outset;"
                                       "border-width: 2px;"
                                       "border-radius: 10px;"
                                       "border-color: rgb(158, 140, 0);"
                                       "font: bold 14px;"
                                       "min-width: 10em;"
                                       "padding: 6px;"
                                       "}"
                                       "QPushButton:pressed{"
                                       "background-color: rgb(158, 140, 0);"
                                       "border-style: inset;"
                                       "}"
                                       )
        self.checkboxImage.move(600, 300)
        self.buttonImage.clicked.connect(self.button_image_push)
        nn_pixmap = QPixmap(__file__ + "/../nn_graph.png")
        nn_pixmap = nn_pixmap.scaled(550, 500, QtCore.Qt.KeepAspectRatio)
        self.nn_info.setPixmap(nn_pixmap)
        self.nn_info.move(50, 350)
        self.nn_accurate.move(650, 350)
        # self.nn_accurate.setText("Точность Модели:   " + str(self.nn_class.info()) + "%")
        self.nn_accurate.setText("Точность Модели:   " + str(77.24) + "%")

    def button_archive_push(self):
        file_dialog = QFileDialog(self)
        if self.checkboxArch.isChecked():
            try:
                file_dir_in = file_dialog.getOpenFileName(self, 'Выберите архив', __file__ + "/../../")[0]
                print(file_dir_in)
                file_dir_out = __file__ + "/../Interface_entry/Not_Sorted.rar"
                os.rename(file_dir_in, file_dir_out)
                patoolib.extract_archive(file_dir_out, outdir=__file__ + "/../Interface_entry/No_Sorted")
                os.remove(file_dir_out)
                self.new_window = context_window(self.nn_class.ret())
                self.new_window.show()
            except FileNotFoundError:
                print("File not found")
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            try:
                file_dir_in = file_dialog.getOpenFileName(self, 'Выберите архив', __file__ + "/../../")[0]
                file_dir_out = __file__ + "/../Interface_entry/Not_Sorted.rar"
                shutil.copyfile(file_dir_in, file_dir_out)
                patoolib.extract_archive(file_dir_out, outdir=__file__ + "/../Interface_entry/No_Sorted")
                os.unlink(file_dir_out)
                self.new_window = context_window(self.nn_class.ret())
                self.new_window.show()
            except FileNotFoundError:
                print("File not found")
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())

    def button_image_push(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter(".jpeg")
        if self.checkboxImage.isChecked():
            try:
                file_dir_in = file_dialog.getOpenFileName(self, 'Выберите изображение', __file__ + "/../../")[
                    0]
                exe = file_dir_in.split('/')[-1]
                file_dir_out = __file__ + "/../Interface_entry/No_Sorted/" + exe
                os.rename(file_dir_in, file_dir_out)
                nn = NeuralNetwork("Interface_entry", "Interface_exit")
                self.new_window = context_window(nn.ret())
                self.new_window.show()
            except FileNotFoundError:
                print("File not found")
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            try:
                file_dir_in = file_dialog.getOpenFileName(self, 'Выберите изображение', __file__ + "/../../")[
                    0]
                exe = file_dir_in.split('/')[-1]
                file_dir_out = __file__ + "/../Interface_entry/No_Sorted/" + exe
                shutil.copyfile(file_dir_in, file_dir_out)
                nn = NeuralNetwork("Interface_entry", "Interface_exit")
                self.new_window = context_window(nn.ret())
                self.new_window.show()
            except FileNotFoundError:
                print("File not found")
            except:
                print("Unexpected error:", sys.exc_info())


def prepare_data(directory, shuffle, batch_size, image_size):  # Подготовка сета
    return image_dataset_from_directory(
        directory,
        shuffle=shuffle,  # рандомизация данных
        batch_size=batch_size,
        image_size=image_size
    )


class NeuralNetwork:
    def __init__(self, in_dir, out_dir):
        # ___________________________________________________________________________________Востановление модели
        # Восстановим в точности ту же модель, включая веса и оптимизатор
        self.predictions = []
        self.files_to_sort = ""
        self.name_file = ""
        self.model = keras.models.load_model('Model_GPU.h5')
        # Покажем архитектуру модели
        self.model.summary()
        self.BATCH_SIZE = 64  # Размер кластера
        self.IMG_SIZE = (256, 256)  # Размер изображения (после реформации)
        self.in_dir = in_dir
        self.out_dir = out_dir

    def info(self):
        test_dataset = prepare_data("test", True, batch_size=self.BATCH_SIZE, image_size=self.IMG_SIZE)
        loss, acc = self.model.evaluate(test_dataset, verbose=2)
        print("Точность восстановленной модели: {:5.2f}%".format(100 * acc))  # определение точности (улучшить)
        return format(acc * 100, '3f')

    def predict(self, in_dir, out_dir="Interface_exit"):
        extra_dataset = prepare_data("Interface_entry", False, self.BATCH_SIZE, self.IMG_SIZE)  # сет для тестов
        predictions = self.model.predict(extra_dataset)  # ______Предсказывание
        files_to_sort = os.listdir(in_dir + "/No_Sorted")
        for el in range(len(predictions)):
            name_file = files_to_sort[el]
            if predictions[el][0] > predictions[el][1]:
                shutil.copyfile(in_dir + "/No_Sorted/" + name_file,
                                out_dir + "/Normal/" + name_file)
            else:
                shutil.copyfile(in_dir + "/No_Sorted/" + name_file,
                                out_dir + "/Pneumonia/" + name_file)
        return predictions

    def ret(self):
        out_arr = []
        for el in self.predict(self.in_dir, self.out_dir):
            arr = []
            for ej in el:
                arr.append(format(ej, '4f'))
            out_arr.append(arr)
        return out_arr


if __name__ == '__main__':
    if len(tf.config.experimental.list_physical_devices('GPU')) != 0:  # _________________ проверка на доступность GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # _______Запуск с использованием видеоядра
    if "Interface_entry" not in os.listdir():
        os.mkdir("Interface_entry")
        os.mkdir("Interface_entry/No_Sorted")
    elif "No_Sorted" not in os.listdir("Interface_entry/"):  # ______________ Проверка на существование системных папок
        os.mkdir("Interface_entry/No_Sorted")
    if "Interface_exit" not in os.listdir():
        os.mkdir("Interface_exit")
        os.mkdir("Interface_exit/Normal")
        os.mkdir("Interface_exit/Pneumonia")
    elif "Normal" not in os.listdir("Interface_exit/"):
        os.mkdir("Interface_exit/Normal")
    elif "Pneumonia" not in os.listdir("Interface_exit/"):
        os.mkdir("Interface_exit/Pneumonia")
    if "Model_GPU.h5" in os.listdir():
        # Создадим класс приложения PyQT
        app = QApplication(sys.argv)
        # А теперь создадим и покажем пользователю экземпляр
        # нашего виджета класса Example
        ex = main_Interface()
        ex.show()
        # Будем ждать, пока пользователь не завершил исполнение QApplication,
        # а потом завершим и нашу программу
        sys.exit(app.exec())
        # os.remove("Interface_entry/No_Sorted")
        # os.remove("Interface_entry")
        # os.remove("Interface_exit/Normal")
        # os.remove("Interface_exit/Pneumonia")
        # os.remove("Interface_exit")
    else:
        # Создадим класс приложения PyQT
        app = QApplication(sys.argv)
        # А теперь создадим и покажем пользователю экземпляр
        # нашего виджета класса Example
        ex = nn_error()
        ex.show()
        # Будем ждать, пока пользователь не завершил исполнение QApplication,
        # а потом завершим и нашу программу
        sys.exit(app.exec())
