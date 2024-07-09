# encoding:utf-8
import sys

import matplotlib.pyplot as plt
import numpy as np
# import cupy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QTableWidgetItem, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter
from PyQt5.QtGui import QGuiApplication
import time
from main_Window import Ui_MainWindow
from Add_test_range_Window import Ui_Dialog

# import cv2
import pandas as pd
from tqdm import tqdm
from DDFS.analytic_method import Resolution as Res_a
from DDFS.kalman_method import Resolution as Res_k
from DDFS.write_root import _write_root
import datetime
from matplotlib.patches import Circle
from DDFS.element import *
import matplotlib
# import uproot3 as uproot
import uproot

plt.rcParams['savefig.dpi'] = 600  # 图像保存时的DPI
# plt.rcParams['figure.dpi'] = 300   # 图形的DPI
# from pycallgraph2 import PyCallGraph
# from pycallgraph2.output import GraphvizOutput


c = 0.299792458e-3

# from multiprocessAccelerate import multi_process

matplotlib.use('Agg')

INFO_TYPE_LAYER = ["Material_budget", "Radius", "Half_z", "Efficiency", "Loc0", "Loc1"]
INFO_TYPE_ENVIRONMENT = ["B"]

INFO_TYPE_PARTICLE = ["Mass", "Charge", "Beam_spot", "Weight"]
INFO_TYPE_PARTICLE_EMIT = ["p", "theta", "phi", "beta"]

X_AXIS_TYPE = ["p", "theta", "phi"]
Y_AXIS_TYPE = ["dr", "dz", "dt", "df", "dp", "dp2", "det", "p", "theta", "phi"]
C_AXIS_TYPE = ["dr", "dz", "dt", "df", "dp", "dp2", "det", "p", "theta", "phi"]

X_AXIS_TYPE_KALMAN = ["p", "theta", "phi"]
Y_AXIS_TYPE_KALMAN = ["dr", "dz", "dt", "df", "dp", "dp2"]
FILTER_TYPE_KALMAN = ["ori", "forward", "backward"]
C_AXIS_TYPE_KALMAN = ["dr", "dz", "dt", "df", "dp", "dp2", "p", "theta", "phi"]

export_order = []

multiprocessAccelerate = False  # 别用，用了更慢 :-(
if multiprocessAccelerate:
    process_num = 2


def runnn(res_a):
    ini_para, ret_para = res_a.analytic_estimate()
    return ini_para, ret_para


class Analytic_cal(QThread):
    finished_signal = pyqtSignal(object, int)
    progress_signal = pyqtSignal(int)

    def __init__(self, dec_info_list, envir_info_list, emit, test_num):
        super().__init__()
        self.dec_info_list = dec_info_list
        self.envir_info_list = envir_info_list
        self.emit = emit
        self.test_num = test_num
        self.multi = multiprocessAccelerate

    def run(self):
        # with PyCallGraph(output=GraphvizOutput(output_file='Analytic.png', output_height=3000, font_size=30)):

        # 执行耗时任务并传递参数
        result = self.submit_calculation(self.dec_info_list, self.envir_info_list, self.emit, self.test_num)
        # 发送任务完成的信号
        self.finished_signal.emit(result, 0)

    def submit_calculation(self, dec_info_list, envir_info_list, emit, test_num):
        # 耗时任务的代码，可以使用参数

        res_a = Res_a(dec_info_list, envir_info_list, emit.copy())

        result = Result(test_num=test_num)

        error_list = {}

        if self.multi == False:
            if test_num > 0:
                for num in tqdm(range(test_num)):
                    # ini_para, ret_para = res_a.analytic_estimate()
                    try:
                        ini_para, ret_para = runnn(res_a)
                    except Exception as e:
                        print(e)
                        error_type = type(e).__name__
                        if error_type in error_list:
                            error_list[error_type] += 1
                        else:
                            error_list[error_type] = 1
                        continue
                    result.append(ini_para, ret_para)
                    self.progress_signal.emit(int((num + 1) / test_num * 100))
            else:
                print("haha")  # 准备做一个死循环采集
        # elif self.multi == True:
        #     if test_num > 0:
        #         result_temp = multi_process(runnn, [(res_a.copy(),) for i in range(test_num)])
        #         for i in result_temp:
        #             result.append(i[0], i[1])
        #
        #     else:
        #         print("haha")

        # result.plot()
        # print(result)
        emit_mode = emit.emit_param["Emit_mode"][0]
        result.set_emit_mode(emit_mode)
        return result, error_list


class Kalman_cal(QThread):
    finished_signal = pyqtSignal(object, int)
    progress_signal = pyqtSignal(int)

    def __init__(self, dec_info_list, envir_info_list, emit, test_num, forward_FLAG, plot_FLAG):
        super().__init__()
        self.dec_info_list = dec_info_list
        self.envir_info_list = envir_info_list
        self.emit = emit
        self.test_num = test_num
        self.multi = multiprocessAccelerate
        self.forward_FLAG = forward_FLAG
        self.plot_FLAG = plot_FLAG

    def run(self):
        # with PyCallGraph(output=GraphvizOutput(output_file='kalman.png', output_height=3000, font_size=30)):

        # 执行耗时任务并传递参数
        result = self.submit_calculation(self.dec_info_list, self.envir_info_list, self.emit, self.test_num)

        # 发送任务完成的信号
        self.finished_signal.emit(result, 1)

    def submit_calculation(self, dec_info_list, envir_info_list, emit, test_num):
        # 耗时任务的代码，可以使用参数

        res_k = Res_k(dec_info_list, envir_info_list, emit.copy())

        fig10 = plt.figure(figsize=(10, 10))
        ax = fig10.add_subplot(231)
        ax2 = fig10.add_subplot(234)
        ax3 = fig10.add_subplot(232)
        ax4 = fig10.add_subplot(233)
        ax5 = fig10.add_subplot(235)
        ax6 = fig10.add_subplot(236)

        step_seed = 1
        mode = emit.get_info("dir")["Emit_mode"][0]
        for key, item in mode.items():
            if item["type"] == "steps":
                step_seed *= item["steps"]

        error_list = {}
        re = Result(test_num=test_num)
        np.random.seed(int(np.fmod(time.time(), 10) * 1e5))
        seed = np.random.randint(0, 100000)
        if test_num > 0:
            for num in tqdm(range(test_num)):
                try:

                    if num % step_seed == 0:
                        np.random.seed(int(np.fmod(time.time(), 10) * 1e5))
                        seed = np.random.randint(0, 100000)

                    np.random.seed(seed)

                    ori_path, observe_store_XYZ = res_k.generate_path()

                    res_k.generate_ref_path()
                    # res_k.initialize_filter(Forward=False)
                    if self.forward_FLAG:
                        forward_path = res_k.forward_kalman_estimate()
                    # else:
                    #     res_k.initialize_filter()

                    # forward_path = res_k.forward_kalman_estimate()
                    backward_path = res_k.backward_kalman_estimate()
                    # res_k.direct_fit()
                    if self.plot_FLAG:
                        ax.plot(ori_path[:, 0], ori_path[:, 1], 'r', alpha=0.5)
                        # ax.plot(res_k.radius_list * np.cos(forward_path[:, 0]),
                        #         res_k.radius_list * np.sin(forward_path[:, 0]), color='g', alpha=0.3)
                        ax.plot(res_k.radius_list * np.cos(backward_path[:, 0]),
                                res_k.radius_list * np.sin(backward_path[:, 0]), color='b', alpha=0.5)
                        ax.plot(observe_store_XYZ[:, 0], observe_store_XYZ[:, 1], 'k', alpha=0.1)
                        #
                        ax2.plot(res_k.radius_list, ori_path[:, 2], 'r', alpha=0.5)
                        # ax2.plot(res_k.radius_list, forward_path[:, 1], color='g', alpha=0.3)
                        ax2.plot(res_k.radius_list, backward_path[:, 1], color='b', alpha=0.5)
                        ax2.plot(res_k.radius_list, observe_store_XYZ[:, 2], 'k', alpha=0.1)

                        ax3.plot(res_k.radius_list,
                                 res_k.radius_list * (res_k.param_helix_store[:, 0] - res_k.observe_store[:, 0]), 'r.',
                                 alpha=0.01)
                        ax3.set_title("ori - error - rphi")

                        ax4.plot(res_k.radius_list, res_k.param_helix_store[:, 1] - res_k.observe_store[:, 1], 'r.',
                                 alpha=0.01)
                        ax4.set_title("ori - error - z")

                        ax5.plot(res_k.radius_list,
                                 res_k.radius_list * (
                                             res_k.kalman_fit_store_backward[:, 0] - res_k.param_helix_store[:, 0]),
                                 'b.',
                                 alpha=0.01)
                        ax5.set_title("backfit - error - rphi")

                        ax6.plot(res_k.radius_list,
                                 res_k.kalman_fit_store_backward[:, 1] - res_k.param_helix_store[:, 1], 'b.',
                                 alpha=0.01)
                        ax6.set_title("backfit - error - z")

                    ini, result = res_k.result_analysis()
                except Exception as e:

                    print(e)
                    error_type = type(e).__name__
                    if error_type in error_list:
                        error_list[error_type] += 1
                    else:
                        error_list[error_type] = 1
                    continue

                re.append(ini, result)

                self.progress_signal.emit(int((num + 1) / test_num * 100))

            rad_list = self.dec_info_list[0]

            for i in rad_list:
                circle = Circle((0., 0.), i, edgecolor='k', facecolor='none', alpha=0.5)
                ax.add_patch(circle)

            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            # ax.grid(True)
            ax.axis('equal')
            ax.set_title("r-phi plane")

            ax2.set_xlabel('r (mm)')
            ax2.set_ylabel('z (mm)')
            ax2.grid(True)
            ax2.set_title("r-z plane")

            ax3.set_xlabel('r (mm)')
            ax3.set_ylabel('r*phi error (mm)')
            ax3.grid(True)

            ax4.set_xlabel('r (mm)')
            ax4.set_ylabel('z error (mm)')
            ax4.grid(True)

            ax5.set_xlabel('r (mm)')
            ax5.set_ylabel('r*phi error (mm)')
            ax5.grid(True)

            ax6.set_xlabel('r (mm)')
            ax6.set_ylabel('z error (mm)')
            ax6.grid(True)



        else:
            print("haha")  # 准备做一个死循环采集
        emit_mode = emit.emit_param["Emit_mode"][0]
        re.set_emit_mode(emit_mode)
        return re, fig10, error_list


class Kalman_post(QThread):
    finished_signal = pyqtSignal(object, int)
    progress_signal = pyqtSignal(int)

    def __init__(self, Kalman_result, emitmode, dec_length):
        super().__init__()
        self.Kalman_result = Kalman_result
        self.emitmode = emitmode
        self.dec_length = dec_length

    def run(self):
        # 执行耗时任务并传递参数

        for item in tqdm(self.Kalman_result.kalman_post_process(self.emitmode, self.dec_length)):
            self.progress_signal.emit(int(item))
        # 发送任务完成的信号
        self.finished_signal.emit("0", 1)


class save_Thread(QThread):
    finished_signal = pyqtSignal(object, int)
    progress_signal = pyqtSignal(int)

    def __init__(self, name, export_table, treename="kalman"):
        super().__init__()
        self.name = name
        self.export_table = export_table
        self.treename = treename

    def run(self):
        # 执行耗时任务并传递参数
        # print("hahahahhah")
        _write_root(self.name, self.export_table, self.treename)
        # print("file saved")

        # 发送任务完成的信号
        self.finished_signal.emit(1, 0)


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, fig=None):
        if fig is None:
            self.fig = Figure(figsize=(5, 4), dpi=300)
            self.axes = self.fig.add_subplot(111)
        else:
            self.fig = fig
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def clear(self):
        self.axes.cla()
        self.draw()

    def update_pic(self, fig):
        self.figure = fig

        self.draw()


class MyPyQT_Form(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyPyQT_Form, self).__init__()

        self.setupUi(self)
        self.thread = {}

        self.layout = QVBoxLayout(self.pic_show)
        self.canvas = MyMplCanvas(self.pic_show)
        self.toolbar = NavigationToolbar(self.canvas, self.pic_show)

        self.layout.addWidget(self.toolbar)

        self.layout.addWidget(self.canvas)

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        try:
            d = Detector()
            d.load_designed("tmp\\detector_designed.csv")
            self.detectors = [d]
            self.update_layer_table()
        except Exception as e:
            self.detectors = [Detector()]
            self.add_layer()

        try:
            emit = Emitter()
            emit.load("tmp\\emitter.json")
            self.emitters = [emit]
        except Exception as e:
            self.emitters = [Emitter()]
            self.add_particle()
            self.emitters[self.Emitter_combox.currentIndex()].update_particle(self.particle_combox.currentIndex(),
                                                                              INFO_TYPE_PARTICLE[3], float(1))

        self.update_particle_table()
        try:
            envir = Environment()
            envir.load("tmp\\environment.json")
            self.environment = [envir]

        except Exception as e:
            self.environment = [Environment()]
        # self.update_envir_table()

        self.startEvent()

    def startEvent(self):
        self.function_page.setCurrentIndex(0)
        self.layer_info.verticalHeader().setVisible(True)
        self.layer_info.horizontalHeader().setVisible(True)
        self.particle_info.verticalHeader().setVisible(True)
        self.particle_info.horizontalHeader().setVisible(True)
        self.par_emit_info.verticalHeader().setVisible(True)
        self.par_emit_info.horizontalHeader().setVisible(True)

        self.submit_method_comboBox.setCurrentIndex(1)
        self.RE_checkbox.setChecked(1)

        self.Result = None

        # self.layer_info.setCurrentCell(1, 1)
        # self.layer_info.item(0, 0).setText(str(1))
        # self.layer_info.setCurrentItem(None)
        # print(self.layer_info.currentItem().text())

        pass

    def closeEvent(self, event):

        reply = QMessageBox.question(self, '退出', '你确认要退出吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        idx = self.detector_comboBox.currentIndex()
        dec = self.detectors[idx]
        dec.store_designed("tmp\\detector_designed.csv")

        emit = self.emitters[self.Emitter_combox.currentIndex()]
        emit.export("tmp\\emitter.json")

        envir = self.environment[0]
        envir.export("tmp\\environment.json")

        if reply == QMessageBox.Yes:
            sys.exit(0)
        else:
            event.ignore()

    def create_condition(self):
        idx_dec = self.detector_comboBox.currentIndex()
        second_window = constrant_setting_Window(self.detectors[idx_dec], 1, 1)
        result = second_window.exec_()
        if result == 1:
            # 如果对话框被接受，获取返回的自定义值
            # returned_value = second_window.custom_value
            print('用户点击了确定按钮，并返回值:', 1)
        elif result == 0:
            print('用户点击了取消按钮')

    def set_page(self):
        page_name = ["Detector Design / Visualization", "Environment / Particle / Testing range",
                     "Submit Calculation / Plot / Export data"]
        page_index = (self.function_page.currentIndex() + 1) % 3
        if page_index == 2:
            self.update_submit_combox()
        self.function_page.setCurrentIndex(page_index)
        self.func_select_button.setText(page_name[page_index])
        pass

    def add_Emitter(self):
        self.emitters.append(Emitter())
        self.Emitter_combox.addItem("Emitter " + str(len(self.emitters)))

        idx = len(self.emitters) - 1
        self.emitters[idx].add_particle(Particle(charge=1, mass=0.106))
        self.Emitter_combox.setCurrentIndex(idx)
        pass

    def add_detector(self):
        self.detectors.append(Detector())
        self.detector_comboBox.addItem("Detector " + str(len(self.detectors)))
        # print(self.detector_comboBox.currentIndex())

        idx = len(self.detectors) - 1
        self.detectors[idx].add_layer(SiLayer())
        self.detector_comboBox.setCurrentIndex(idx)
        pass

    def add_particle(self):
        idx = self.Emitter_combox.currentIndex()

        self.emitters[idx].add_particle(Particle(charge=-1, mass=0.106))

        self.update_particle_combox()

        # print(len(self.emitters[idx]))
        # self.particle_combox.addItem("Particle " + str(len(self.emitters[idx])))

        idx_par = len(self.emitters[idx]) - 1
        self.particle_combox.setCurrentIndex(idx_par)

        self.update_particle_table()
        pass

    def add_layer(self):
        idx = self.detector_comboBox.currentIndex()

        self.detectors[idx].add_layer(SiLayer())
        self.update_layer_table()
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_layer()

        elif event.key() == Qt.Key_Escape:
            self.close()

        # shift + "+"
        elif event.key() == Qt.Key_Plus and self.function_page.currentIndex() == 0:
            self.add_layer()

    def delete_layer(self):
        if self.layer_info.currentItem() is None:
            return False
        layer_idx = self.layer_info.currentItem().row()

        dec_idx = self.detector_comboBox.currentIndex()


        self.detectors[dec_idx].delete_layer(layer_idx)
        self.update_layer_table()
        pass

    def update_particle_combox(self):

        idx_emi = self.Emitter_combox.currentIndex()
        idx_par = self.particle_combox.currentIndex()

        if len(self.emitters[idx_emi]) <= idx_par:
            idx_par = len(self.emitters[idx_emi]) - 1

        self.particle_combox.clear()

        for i in range(len(self.emitters[idx_emi])):
            self.particle_combox.addItem("Particle " + str(i + 1))

        # print("idx_emi", idx_emi, "adx_par", idx_par)
        self.particle_combox.setCurrentIndex(idx_par)

    def update_particle_info(self, item):
        if self.particle_info.currentItem() is None:
            return 0
        if self.particle_info.currentItem().row() == item.row() and self.particle_info.currentItem().column() == item.column():
            self.emitters[self.Emitter_combox.currentIndex()].update_particle(self.particle_combox.currentIndex(),
                                                                              INFO_TYPE_PARTICLE[item.column()], float(
                    self.particle_info.item(item.row(), item.column()).text()))
            # idx = self.detector_comboBox.currentIndex()

            self.update_particle_table()
            return 1

    # def update_particle_info(self, item):
    #     if self.particle_info.currentItem() is None:
    #         return 0
    #     if self.particle_info.currentItem().row() == item.row() and self.particle_info.currentItem().column() == item.column():
    #         self.emitters[self.Emitter_combox.currentIndex()].update_particle(self.particle_combox.currentIndex(),
    #                                                                           INFO_TYPE_PARTICLE[item.column()], float(
    #                 self.particle_info.item(item.row(), item.column()).text()))
    #         # idx = self.detector_comboBox.currentIndex()
    #
    #         self.update_particle_table()
    #         return 1

    def generate_emit_mode(self):
        dic = {}
        mode_temp = self.emitters[self.Emitter_combox.currentIndex()].get_info("dir")["Emit_mode"][
            self.particle_combox.currentIndex()]
        for i in range(0, 3):
            item = self.par_emit_info.item(i, 2)

            emit_type = item.text()
            dic_temp = {}

            if emit_type == "even":
                item0 = self.par_emit_info.item(i, 1)
                if item0.text() == '':
                    maxvalue = 0
                else:
                    maxvalue = float(item0.text())

                item0 = self.par_emit_info.item(i, 0)
                if item0.text() == '':
                    minvalue = 0
                else:
                    minvalue = float(item0.text())

                dic_temp["type"] = emit_type
                dic_temp["maxvalue"] = maxvalue
                dic_temp["minvalue"] = minvalue
            elif emit_type == "fixed":
                item0 = self.par_emit_info.item(i, 0)
                value = float(item0.text())

                dic_temp["type"] = emit_type
                dic_temp["value"] = value
            elif emit_type == "steps":
                item0 = self.par_emit_info.item(i, 1)
                if item0.text() == '':
                    maxvalue = 0
                else:
                    maxvalue = float(item0.text())

                item0 = self.par_emit_info.item(i, 0)
                if item0.text() == '':
                    minvalue = 0
                else:
                    minvalue = float(item0.text())

                item0 = self.par_emit_info.item(i, 3)
                if item0.text() == '':
                    steps = 0
                else:
                    steps = float(item0.text())

                dic_temp["type"] = emit_type
                dic_temp["maxvalue"] = maxvalue
                dic_temp["minvalue"] = minvalue
                dic_temp["steps"] = int(steps) if steps > 1 else 2
                dic_temp["count"] = 0
            else:
                self.show_error_message("wrong type, ensure the type is 'even', 'fixed' or 'steps'")
                return mode_temp
                # raise TypeError("wrong type")

            dic[INFO_TYPE_PARTICLE_EMIT[i]] = dic_temp
        return dic

    def update_particle_emit_info(self, item):
        if self.par_emit_info.currentItem() is None:
            return 0
        if self.par_emit_info.currentItem().row() == item.row() and self.par_emit_info.currentItem().column() == item.column():
            emit_mode = self.generate_emit_mode()
            self.emitters[self.Emitter_combox.currentIndex()].update_emit_param(self.particle_combox.currentIndex(),
                                                                                emit_mode)
            # idx = self.detector_comboBox.currentIndex()

            self.update_particle_table()

        return 1

    def update_layer_info(self, item):
        if self.layer_info.currentItem() is None:
            return 0
        if self.layer_info.currentItem().row() == item.row() and self.layer_info.currentItem().column() == item.column():
            try:
                self.detectors[self.detector_comboBox.currentIndex()].update_layer(item.row(),
                                                                                   INFO_TYPE_LAYER[item.column()], float(
                        self.layer_info.item(item.row(), item.column()).text()))
                idx = self.detector_comboBox.currentIndex()

                self.update_layer_table()
                return 1

            except Exception as e:
                self.show_error_message("Please input a number")
                self.update_layer_table()
                return 0

    def update_particle_table(self):
        # print(self.detectors)

        idx_emit = self.Emitter_combox.currentIndex()
        # self.update_particle_combox()
        idx_par = self.particle_combox.currentIndex()

        emitter = self.emitters[idx_emit]

        data_all_particle = emitter.get_info("dir")

        par_info = data_all_particle["Particle_classes"][idx_par].get_info("dir")
        par_weight = data_all_particle["Weight"][idx_par]
        par_emit_info = data_all_particle["Emit_mode"][idx_par]

        self.particle_info.setCurrentItem(None)

        for i, key in enumerate(par_info):
            if not self.particle_info.item(0, i):
                self.particle_info.setItem(0, i, QTableWidgetItem())
            item = self.particle_info.item(0, i)
            item.setText(str(par_info[key]))

        if not self.particle_info.item(0, 3):
            self.particle_info.setItem(0, 3, QTableWidgetItem())
        item = self.particle_info.item(0, 3)
        item.setText(str(par_weight))

        for i, key in enumerate(par_emit_info):
            if par_emit_info[key]["type"] == "even":
                for j in range(0, 4):
                    if not self.par_emit_info.item(i, j):
                        self.par_emit_info.setItem(i, j, QTableWidgetItem())

                item = self.par_emit_info.item(i, 0)
                item.setText(str(par_emit_info[key]["minvalue"]))
                item = self.par_emit_info.item(i, 1)
                item.setText(str(par_emit_info[key]["maxvalue"]))
                item = self.par_emit_info.item(i, 2)
                item.setText("even")
                item = self.par_emit_info.item(i, 3)
                item.setText(None)

            elif par_emit_info[key]["type"] == "fixed":
                for j in range(0, 4):
                    if not self.par_emit_info.item(i, j):
                        self.par_emit_info.setItem(i, j, QTableWidgetItem())
                item = self.par_emit_info.item(i, 0)
                item.setText(str(par_emit_info[key]["value"]))
                item = self.par_emit_info.item(i, 1)
                item.setText(None)
                item = self.par_emit_info.item(i, 2)
                item.setText("fixed")
                item = self.par_emit_info.item(i, 3)
                item.setText(None)

            elif par_emit_info[key]["type"] == "steps":
                for j in range(0, 4):
                    if not self.par_emit_info.item(i, j):
                        self.par_emit_info.setItem(i, j, QTableWidgetItem())

                item = self.par_emit_info.item(i, 0)
                item.setText(str(par_emit_info[key]["minvalue"]))
                item = self.par_emit_info.item(i, 1)
                item.setText(str(par_emit_info[key]["maxvalue"]))
                item = self.par_emit_info.item(i, 2)
                item.setText("steps")
                item = self.par_emit_info.item(i, 3)
                item.setText(str(par_emit_info[key]["steps"]))



            else:
                raise TypeError("mode is wrong")

        print(self.emitters[idx_emit])
        print(data_all_particle["Particle_classes"][idx_par])

        # self.layer_info.setRowCount(len(data_all_layer))
        # for i in range(0, len(data_all_layer)):
        #     if not self.layer_info.verticalHeaderItem(i):
        #         self.layer_info.setVerticalHeaderItem(i, QTableWidgetItem())
        #     item = self.layer_info.verticalHeaderItem(i)
        #     item.setText("Layer " + str(i))
        #
        # self.layer_info.setColumnCount(7)
        # for idx_emit, line in enumerate(data_all_layer):
        #     for idy, row in enumerate(line):
        #
        #         if not self.layer_info.item(idx_emit, idy):
        #             self.layer_info.setItem(idx_emit, idy, QTableWidgetItem())
        #         item = self.layer_info.item(idx_emit, idy)
        #         item.setText(str(row))
        # # print(data_all_layer)
        # pass

    def update_layer_table(self):
        # print(self.detectors)
        idx = self.detector_comboBox.currentIndex()
        print(self.detectors[idx])

        self.layer_info.setCurrentItem(None)
        detector = self.detectors[idx]
        data_all_layer = detector.get_info("list")
        self.layer_info.setRowCount(len(data_all_layer))
        for i in range(0, len(data_all_layer)):
            if not self.layer_info.verticalHeaderItem(i):
                self.layer_info.setVerticalHeaderItem(i, QTableWidgetItem())
            item = self.layer_info.verticalHeaderItem(i)
            item.setText("Layer " + str(i))

        self.layer_info.setColumnCount(7)
        for idx, line in enumerate(data_all_layer):
            for idy, row in enumerate(line):

                if not self.layer_info.item(idx, idy):
                    self.layer_info.setItem(idx, idy, QTableWidgetItem())
                item = self.layer_info.item(idx, idy)
                item.setText(str(round(row, 4)))

        self.visualize_dectetor()

    def update_envir_info(self):
        RE = bool(self.RE_checkbox.checkState())
        MS = bool(self.MS_checkbox.checkState())
        if not self.environment_tableWidget.item(0, 0):
            self.environment_tableWidget.setItem(0, 0, QTableWidgetItem())
        item = self.environment_tableWidget.item(0, 0)
        B = float(item.text())

        self.environment[0].update_environment("B", B)
        self.environment[0].update_environment("position_resolution", RE)
        self.environment[0].update_environment("multiple_scattering", MS)
        self.update_envir_table()

    def update_envir_table(self):
        if not self.environment_tableWidget.item(0, 0):
            self.environment_tableWidget.setItem(0, 0, QTableWidgetItem())

        item = self.environment_tableWidget.item(0, 0)
        item.setText(str(self.environment[0].get_info("dir")["B"]))
        print(self.environment[0])

    def visualize_dectetor(self):
        idx = self.detector_comboBox.currentIndex()
        detector = self.detectors[idx]
        fig, ax = detector.visualize_detector()

        self.update_pic(fig)

        # print(self.x(), self.y(), self.width(), self.height())

    def load_design(self):
        # Read the CSV file into a DataFrame
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "csv Files (*.csv)",
                                                   options=options)

        if file_path:
            print("选择的文件路径:", file_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(e)
            return -1
        #
        # # Display the DataFrame
        # print(df)
        #
        # radius_list = df['Radius'].tolist()
        # budget_list = df['Budget'].tolist()
        # half_z_list = df['Half_z'].tolist()
        # loc0 = df['Location 0'].tolist()
        # loc1 = df['Location 1'].tolist()
        # effi = df['Efficiency'].tolist()
        try:
            self.add_detector()
            idx = self.detector_comboBox.currentIndex()
            dec = self.detectors[idx]
            dec.load_designed(file_path)

            # dec.delete_layer(0)
            #
            # for i in range(0, len(radius_list)):
            #     dec.add_layer(
            #         SiLayer(material_budget=budget_list[i], radius=radius_list[i], efficiency=effi[i],
            #                 loc0=loc0[i], loc1=loc1[i], half_z=half_z_list[i]))
            self.update_layer_table()
        except Exception as e:

            return -1

        pass

    def export_design(self):
        idx = self.detector_comboBox.currentIndex()
        detector = self.detectors[idx]
        data_all_layer = detector.get_info("list")

        print(data_all_layer)

        data_all_layer = np.array(data_all_layer)

        material_budget = data_all_layer[:, 0]
        radius_list = data_all_layer[:, 1]
        half_z_list = data_all_layer[:, 2]
        efficiency_list = data_all_layer[:, 3]
        loc0_list = data_all_layer[:, 4]
        loc1_list = data_all_layer[:, 5]

        # sys.exit(0)

        # Create a DataFrame from the given data
        data = {
            'Radius': radius_list,
            'Budget': material_budget,
            'Half_z': half_z_list,
            'Location 0': loc0_list,
            'Location 1': loc1_list,
            'Efficiency': efficiency_list
        }

        df = pd.DataFrame(data)

        # 格式化时间作为文件名
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "选择文件路径", "", "csv Files (*.csv)",
                                                  options=options)

        # Export the data to a CSV file
        if fileName != '':
            csvname = fileName + ".csv"
            df.to_csv(csvname, index=False)
        pass

    def update_submit_combox(self):
        idx_dec = self.detector_comboBox.currentIndex()
        idx_emit = self.Emitter_combox.currentIndex()

        self.submit_dec_comboBox.clear()
        self.submit_emit_comboBox.clear()

        for i in range(len(self.detectors)):
            self.submit_dec_comboBox.addItem("Detector " + str(i + 1))

        for i in range(len(self.emitters)):
            self.submit_emit_comboBox.addItem("Emitter " + str(i + 1))

        self.submit_dec_comboBox.setCurrentIndex(idx_dec)
        self.submit_emit_comboBox.setCurrentIndex(idx_emit)

        pass

    def check_submit_info(self):
        pass

    def method_changed(self):
        idx = self.submit_method_comboBox.currentIndex()
        if idx == 0:
            self.two_dir_checkBox.setChecked(0)
            self.two_dir_checkBox.setDisabled(1)
        else:
            self.two_dir_checkBox.setDisabled(0)

    def submit_calculation(self):
        self.check_submit_info()

        dec = self.detectors[self.submit_dec_comboBox.currentIndex()]
        emit = self.emitters[self.submit_emit_comboBox.currentIndex()]
        envir = self.environment[0]

        dec_info_list = dec.get_param()
        envir_info_list = envir.get_param()
        t = time.time()

        test_method_idx = self.submit_method_comboBox.currentIndex()
        test_num = self.test_num_spinBox.value()

        if test_method_idx == 0:
            self.two_dir_checkBox.setChecked(0)
            self.two_dir_checkBox.setDisabled(1)
            print("Analytic")

            self.worker_thread = Analytic_cal(dec_info_list, envir_info_list, emit, test_num)
            self.worker_thread.finished_signal.connect(self.handle_task_completion)
            self.worker_thread.progress_signal.connect(self.update_progress)
            self.submitButton.setEnabled(0)
            self.worker_thread.start()





        elif test_method_idx == 1:
            self.two_dir_checkBox.setEnabled(1)
            # self.two_dir_checkBox.setChecked(1)
            two_dir_FLAG = self.two_dir_checkBox.isChecked()
            plot_FLAG = self.result_visual_checkBox.isChecked()
            print("Kalman")
            self.worker_thread = Kalman_cal(dec_info_list, envir_info_list, emit, test_num, two_dir_FLAG, plot_FLAG)
            self.worker_thread.finished_signal.connect(self.handle_task_completion)
            self.worker_thread.progress_signal.connect(self.update_progress)

            self.submitButton.setEnabled(0)
            self.worker_thread.start()

    def handle_task_completion(self, result, method):
        # 在主线程中处理任务完成的操作，并使用结果
        # print("Task completed with result:", result)
        self.Result = result
        self.Result_Flag = method
        self.current_mode = self.generate_emit_mode()
        self.submitButton.setEnabled(1)

        if method == 0:  # analytic
            # self.canvas.update_pic(self.Result)
            # 创建第二个 fig 对象
            self.plotWidgeth.setCurrentIndex(0)
            error_list = self.Result[1]

            if error_list:
                error_string = "\n".join(
                    [f"发生了 {count} 次 {error_type} 错误。" for error_type, count in error_list.items()])
                # print(error_string)
                self.show_error_message(error_string)
            x_idx = self.dataType_comboBox_X.currentIndex() if self.dataType_comboBox_X.currentIndex() != 0 else 1
            self.dataType_comboBox_X.setCurrentIndex(x_idx)
            # self.Result = self.Result[0]
            self.result_plot()

            self.ResultWidget.setCurrentIndex(0)

        elif method == 1:  # kalman
            self.plotWidgeth.setCurrentIndex(1)
            self.ResultWidget.setCurrentIndex(1)
            self.post_process_thread = Kalman_post(self.Result[0], self.current_mode,
                                                   len(self.detectors[self.detector_comboBox.currentIndex()]))
            self.post_process_thread.progress_signal.connect(self.update_progress)
            self.post_process_thread.finished_signal.connect(self.set_kalman_button_enable)

            self.dataType_comboBox_C_kalman.setCurrentIndex(0)
            self.dataType_comboBox_X_kalman.setCurrentIndex(0)
            self.dataType_comboBox_Y_kalman.setCurrentIndex(0)
            self.dataType_comboBox_p_step.setCurrentIndex(0)
            self.dataType_comboBox_theta_step.setCurrentIndex(0)
            self.dataType_comboBox_phi_step.setCurrentIndex(0)
            self.dataType_comboBox_layer.setCurrentIndex(0)
            self.dataType_comboBox_filter.setCurrentIndex(0)

            self.dataType_comboBox_C_kalman.setEnabled(0)
            self.dataType_comboBox_X_kalman.setEnabled(0)
            self.dataType_comboBox_Y_kalman.setEnabled(0)
            self.dataType_comboBox_p_step.setEnabled(0)
            self.dataType_comboBox_theta_step.setEnabled(0)
            self.dataType_comboBox_phi_step.setEnabled(0)
            self.dataType_comboBox_layer.setEnabled(0)
            self.dataType_comboBox_filter.setEnabled(0)

            self.post_process_thread.start()

            comboxs = {
                "p": self.dataType_comboBox_p_step,
                "theta": self.dataType_comboBox_theta_step,
                "phi": self.dataType_comboBox_phi_step
            }

            if self.Result[0].judge_mode(self.current_mode):
                for key, value in self.current_mode.items():
                    if self.current_mode[key]["type"] == "steps":
                        comboxs[key].clear()
                        comboxs[key].addItem("None")
                        minvalue = self.current_mode[key]["minvalue"]
                        maxvalue = self.current_mode[key]["maxvalue"]
                        for i in np.linspace(minvalue, maxvalue, self.current_mode[key]["steps"]):
                            comboxs[key].addItem(str(i))
                    else:
                        comboxs[key].clear()
                        comboxs[key].addItem("None")
            else:
                for key, value in comboxs.items():
                    comboxs[key].clear()
                    comboxs[key].addItem("None")

            self.dataType_comboBox_layer.clear()
            self.dataType_comboBox_layer.addItem("None")

            for i in range(len(self.detectors[self.detector_comboBox.currentIndex()])):
                self.dataType_comboBox_layer.addItem(str(i + 1))
            self.dataType_comboBox_layer.setCurrentIndex(1)

            error_list = self.Result[2]

            if error_list:
                error_string = "\n".join(
                    [f"发生了 {count} 次 {error_type} 错误。" for error_type, count in error_list.items()])
                # print(error_string)
                self.show_error_message(error_string)
            if self.result_visual_checkBox.checkState() == 2:
                self.update_pic(self.Result[1])
            else:
                fig = plt.figure()
                self.update_pic(fig)

            # self.result_plot()

    def export_result(self):
        if self.Result is None:
            return -1

        export_table = {}
        type = ''
        if self.Result_Flag == 0:
            type = 'analytic'
            state_now = [
                self.dr_checkBox.checkState(),
                self.dz_checkBox.checkState(),
                self.dtheta_checkBox.checkState(),
                self.df_checkBox.checkState(),
                self.dpp_checkBox.checkState(),
                self.dpp2_checkBox.checkState(),
                self.det_checkBox.checkState(),
            ]

            export_table = self.Result[0].initial

            if state_now[0] == 2:
                export_table["dr"] = self.Result[0].get("dr")

            if state_now[1] == 2:
                export_table["dz"] = self.Result[0].get("dz")

            if state_now[2] == 2:
                export_table["dt"] = self.Result[0].get("dt")

            if state_now[3] == 2:
                export_table["df"] = self.Result[0].get("df")

            if state_now[4] == 2:
                export_table["dp"] = self.Result[0].get("dp")

            if state_now[5] == 2:
                export_table["dp2"] = self.Result[0].get("dp2")

            if state_now[6] == 2:
                export_table["det"] = self.Result[0].get("det")

            # print(export_table)

            print("analytic result")
            pass
        elif self.Result_Flag == 1:
            type = 'kalman'
            state_now = [
                self.ori_path_checkBox.checkState(),
                self.forward_path_checkBox.checkState(),
                self.backward_path_checkBox.checkState(),
                self.ob_path_checkBox.checkState(),
                self.res_chi2_checkBox.checkState(),
            ]

            export_table = self.Result[0].initial

            export_table["forward_dr"] = self.Result[0].get("forward_dr")
            export_table["forward_dz"] = self.Result[0].get("forward_dz")
            export_table["forward_dt"] = self.Result[0].get("forward_dt")
            export_table["forward_df"] = self.Result[0].get("forward_df")
            export_table["forward_dp"] = self.Result[0].get("forward_dp")
            export_table["forward_dp2"] = self.Result[0].get("forward_dp2")

            export_table["backward_dr"] = self.Result[0].get("forward_dr")
            export_table["backward_dz"] = self.Result[0].get("backward_dz")
            export_table["backward_dt"] = self.Result[0].get("backward_dt")
            export_table["backward_df"] = self.Result[0].get("backward_df")
            export_table["backward_dp"] = self.Result[0].get("backward_dp")
            export_table["backward_dp2"] = self.Result[0].get("backward_dp2")

            if state_now[0] == 2:
                export_table["ori_path"] = self.Result[0].get("ori_path")

            if state_now[1] == 2:
                export_table["f_dr_first"] = self.Result[0].get("f_dr_first")
                export_table["f_dz_first"] = self.Result[0].get("f_dz_first")
                export_table["f_dt_first"] = self.Result[0].get("f_dt_first")
                export_table["f_df_first"] = self.Result[0].get("f_df_first")
                export_table["f_dp_first"] = self.Result[0].get("f_dp_first")
                export_table["f_dp2_first"] = self.Result[0].get("f_dp2_first")

            if state_now[2] == 2:
                export_table["b_dr_first"] = self.Result[0].get("b_dr_first")
                export_table["b_dz_first"] = self.Result[0].get("b_dz_first")
                export_table["b_dt_first"] = self.Result[0].get("b_dt_first")
                export_table["b_df_first"] = self.Result[0].get("b_df_first")
                export_table["b_dp_first"] = self.Result[0].get("b_dp_first")
                export_table["b_dp2_first"] = self.Result[0].get("b_dp2_first")

            if state_now[3] == 2:
                export_table["measure_path"] = self.Result[0].get("measure_path")

            if state_now[4] == 2:
                export_table["res_ori"] = self.Result[0].get("res_ori")
                export_table["res_forward"] = self.Result[0].get("res_forward")
                export_table["res_backward"] = self.Result[0].get("res_backward")
                export_table["chi2_forward"] = self.Result[0].get("chi2_forward")
                export_table["chi2_backward"] = self.Result[0].get("chi2_backward")

            # export_table["emit_mode"] = self.Result[0].emit_mode

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "选择文件路径", "", "root Files (*.root)",
                                                  options=options)

        # Export the data to a root file
        if fileName != '':
            rootname = fileName
            json.dump(self.Result[0].emit_mode, open(rootname + "_" + type + '.json', 'w'))

            name = rootname + '.root'

            print(name)
            self.save_thread = save_Thread(name, export_table, treename=type)
            self.save_thread.start()
            if self.Result_Flag == 1:
                name2 = rootname + '_kalman_post.root'
                export_table_post = {**self.Result[0].post_initial, **self.Result[0].post_result}
                self.save_thread2 = save_Thread(name2, export_table_post, treename='kalman_post')
                self.save_thread2.start()


        else:
            return -1

    def load_result(self):

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "root Files (*.root)",
                                                   options=options)

        # re = Result(1000)

        re = Result()
        tree_type = re.load_root(file_path)
        print("treee", tree_type)

        self.current_mode = self.generate_emit_mode()
        self.submitButton.setEnabled(1)

        # self.handle_task_completion(self.Result, 0)

        # self.handle_task_completion(self.Result, 1)

        if tree_type == "analytic":
            self.Result_Flag = 0
            self.current_mode = re.emit_mode

            self.Result = [re, {}]  # analytic
            # self.canvas.update_pic(self.Result)
            # 创建第二个 fig 对象
            self.plotWidgeth.setCurrentIndex(0)
            error_list = self.Result[1]

            if error_list:
                error_string = "\n".join(
                    [f"发生了 {count} 次 {error_type} 错误。" for error_type, count in error_list.items()])
                # print(error_string)
                self.show_error_message(error_string)
            x_idx = self.dataType_comboBox_X.currentIndex() if self.dataType_comboBox_X.currentIndex() != 0 else 1
            self.dataType_comboBox_X.setCurrentIndex(x_idx)
            # self.Result = self.Result[0]
            self.result_plot()

            self.ResultWidget.setCurrentIndex(0)


        elif tree_type == "kalman" or tree_type == "kalman_post":
            self.Result_Flag = 1

            self.Result = [re, plt.figure(), {}]
            self.plotWidgeth.setCurrentIndex(1)
            self.ResultWidget.setCurrentIndex(1)

            self.current_mode = re.emit_mode

            # self.current_mode = json.load(json_path + "_" + tree_type + '.json')

            # self.dataType_comboBox_C_kalman.setCurrentIndex(0)
            # self.dataType_comboBox_X_kalman.setCurrentIndex(0)
            # self.dataType_comboBox_Y_kalman.setCurrentIndex(0)
            # self.dataType_comboBox_p_step.setCurrentIndex(0)
            # self.dataType_comboBox_theta_step.setCurrentIndex(0)
            # self.dataType_comboBox_phi_step.setCurrentIndex(0)
            # self.dataType_comboBox_layer.setCurrentIndex(0)
            # self.dataType_comboBox_filter.setCurrentIndex(0)
            #
            # self.dataType_comboBox_C_kalman.setEnabled(0)
            # self.dataType_comboBox_X_kalman.setEnabled(0)
            # self.dataType_comboBox_Y_kalman.setEnabled(0)
            # self.dataType_comboBox_p_step.setEnabled(0)
            # self.dataType_comboBox_theta_step.setEnabled(0)
            # self.dataType_comboBox_phi_step.setEnabled(0)
            # self.dataType_comboBox_layer.setEnabled(0)
            # self.dataType_comboBox_filter.setEnabled(0)

            comboxs = {
                "p": self.dataType_comboBox_p_step,
                "theta": self.dataType_comboBox_theta_step,
                "phi": self.dataType_comboBox_phi_step
            }

            if self.Result[0].judge_mode(self.current_mode):
                for key, value in self.current_mode.items():
                    if self.current_mode[key]["type"] == "steps":
                        comboxs[key].clear()
                        comboxs[key].addItem("None")
                        minvalue = self.current_mode[key]["minvalue"]
                        maxvalue = self.current_mode[key]["maxvalue"]
                        for i in np.linspace(minvalue, maxvalue, self.current_mode[key]["steps"]):
                            comboxs[key].addItem(str(i))
                    else:
                        comboxs[key].clear()
                        comboxs[key].addItem("None")
            else:
                for key, value in comboxs.items():
                    comboxs[key].clear()
                    comboxs[key].addItem("None")

            self.dataType_comboBox_layer.clear()
            self.dataType_comboBox_layer.addItem("None")

            detector_length = self.Result[0].post_result["backward_dr"].shape[1]

            for i in range(detector_length):
                self.dataType_comboBox_layer.addItem(str(i + 1))
            self.dataType_comboBox_layer.setCurrentIndex(1)
            #
            # error_list = self.Result[2]
            #
            # if error_list:
            #     error_string = "\n".join(
            #         [f"发生了 {count} 次 {error_type} 错误。" for error_type, count in error_list.items()])
            #     # print(error_string)
            #     self.show_error_message(error_string)
            # if self.result_visual_checkBox.checkState() == 2:
            #     self.update_pic(self.Result[1])
            # else:
            #     fig = plt.figure()
            #     self.update_pic(fig)

            # self.result_plot()

        pass

    def set_kalman_button_enable(self):
        # self.dataType_comboBox_C_kalman.setEnabled(1)
        self.dataType_comboBox_X_kalman.setEnabled(1)
        self.dataType_comboBox_Y_kalman.setEnabled(1)
        self.dataType_comboBox_p_step.setEnabled(1)
        self.dataType_comboBox_theta_step.setEnabled(1)
        self.dataType_comboBox_phi_step.setEnabled(1)
        self.dataType_comboBox_layer.setEnabled(1)
        self.dataType_comboBox_filter.setEnabled(1)

    def update_progress(self, value):
        # 更新进度条
        # print(value)
        self.progressBar.setValue(value)

    def result_plot(self):
        try:
            RE_FLAG = self.Result_Flag
        except Exception as e:
            return -1

        # print("plotting")
        if RE_FLAG == 0 and len(self.Result[0]) != 0:
            # print(len(self.Result[0]))
            X_idx = self.dataType_comboBox_X.currentIndex()
            Y_idx = self.dataType_comboBox_Y.currentIndex()
            c_idx = self.dataType_comboBox_C.currentIndex()
            # print(X_idx, Y_idx, c_idx)
            if X_idx == 0:
                return -1

            X = X_AXIS_TYPE[X_idx - 1]

            if c_idx == 0:
                C = None
            else:
                C = C_AXIS_TYPE[c_idx - 1]

            if Y_idx != 0:
                Y = Y_AXIS_TYPE[Y_idx - 1]

                fig = plt.figure()
                ax = fig.add_subplot(111)
                self.Result[0].analytic_plot(X, Y, colormap=C, emit_mode=self.current_mode)

                self.update_pic(fig)
            else:

                fig = plt.figure()
                for i in range(0, 6):
                    Y_idx = i + 1
                    Y = Y_AXIS_TYPE[Y_idx - 1]
                    ax = fig.add_subplot(2, 3, i + 1)
                    self.Result[0].analytic_plot(X, Y, colormap=C, emit_mode=self.current_mode)
                    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
                    # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))
                self.update_pic(fig)
            pass

        elif RE_FLAG == 1:
            fig = plt.figure()

            X_idx = self.dataType_comboBox_X_kalman.currentIndex()
            Y_idx = self.dataType_comboBox_Y_kalman.currentIndex()
            c_idx = self.dataType_comboBox_C_kalman.currentIndex()

            p_step_idx = None
            theta_step_idx = None
            phi_step_idx = None

            layer_idx = self.dataType_comboBox_layer.currentIndex()
            filter_idx = FILTER_TYPE_KALMAN[
                self.dataType_comboBox_filter.currentIndex() - 1] if self.dataType_comboBox_filter.currentIndex() > 0 else None

            if X_idx == 0 or Y_idx == 0:
                return -1
            X = X_AXIS_TYPE_KALMAN[X_idx - 1]
            Y = Y_AXIS_TYPE_KALMAN[Y_idx - 1]

            if c_idx == 0:
                C = None
            else:
                C = C_AXIS_TYPE_KALMAN[c_idx - 1]

            if self.Result[0].judge_mode(self.current_mode):
                if self.dataType_comboBox_p_step.count() > 1 and self.dataType_comboBox_p_step.currentIndex() > 0:
                    p_step_idx = self.dataType_comboBox_p_step.currentIndex()
                if self.dataType_comboBox_theta_step.count() > 1 and self.dataType_comboBox_theta_step.currentIndex() > 0:
                    theta_step_idx = self.dataType_comboBox_theta_step.currentIndex()
                if self.dataType_comboBox_phi_step.count() > 1 and self.dataType_comboBox_phi_step.currentIndex() > 0:
                    phi_step_idx = self.dataType_comboBox_phi_step.currentIndex()

            self.Result[0].kalman_plot(x_axis=X, y_axis=Y, colormap=C, emit_mode=self.current_mode, p_step=p_step_idx,
                                       theta_step=theta_step_idx, phi_step=phi_step_idx, layer_idx=layer_idx,
                                       filter=filter_idx)
            self.update_pic(fig)

        pass

    def update_pic(self, fig):
        self.layout.removeWidget(self.canvas)
        self.canvas.deleteLater()
        self.layout.removeWidget(self.toolbar)
        self.toolbar.deleteLater()

        self.canvas = MyMplCanvas(self.pic_show, fig)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.canvas)

        self.layout.addWidget(self.toolbar)

        # +62 + 14
        # plt.cla()
        plt.close("all")

        # self.resize(self.width(), self.height() + 1)
        # self.resize(self.width(), self.height() - 1)

        # self.setGeometry(self.x() + 2, self.y() + 62, self.width(), self.height() + 1)
        # self.setGeometry(self.x(), self.y(), self.width(), self.height() - 1)
        pass

    def show_error_message(self, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setText(message)
        error_box.setWindowTitle("错误")
        error_box.exec_()


class constrant_setting_Window(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, detector, emitter, environment):
        super(constrant_setting_Window, self).__init__()
        self.setupUi(self)

        self.detector = detector
        self.emitter = emitter
        self.environment = environment

        self.update_layer_table()

    def update_layer_table(self):
        detector = self.detector

        self.layer_info.setCurrentItem(None)
        data_all_layer = detector.get_info("list")
        self.layer_info.setRowCount(len(data_all_layer))
        for i in range(0, len(data_all_layer)):
            if not self.layer_info.verticalHeaderItem(i):
                self.layer_info.setVerticalHeaderItem(i, QTableWidgetItem())
            item = self.layer_info.verticalHeaderItem(i)
            item.setText("Layer " + str(i))

        self.layer_info.setColumnCount(7)
        for idx, line in enumerate(data_all_layer):
            for idy, row in enumerate(line):

                if not self.layer_info.item(idx, idy):
                    self.layer_info.setItem(idx, idy, QTableWidgetItem())
                item = self.layer_info.item(idx, idy)
                item.setText(str(round(row, 4)))

        self.layer_comboBox.clear()

        for i in range(len(self.detector)):
            self.layer_comboBox.addItem("Layer " + str(i + 1))


if __name__ == '__main__':
    ######使用下面的方式一定程度上可以解决界面模糊问题--解决电脑缩放比例问题
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # main.manual_con
    # trol()

    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())
# pyinstaller -F D:\files\pyproj\dec_use\example\GUI\demo.py
# pyinstaller -F -w demo.py
# pyinstaller --path D:\file\pyproj\sch_new\venv\Lib\site-packages\PyQt5\Qt\bin -F D:\files\pyproj\dec_use\example\GUI\demo.py
# cxfreeze demo.py --target-dir dist
