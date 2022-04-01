# QToolBar, QStatusBar

from PyQt5.QtCore import Qt, QTimer, QSize  # , QPointF, QPoint
from PyQt5.QtGui import QPainter, QPen, QImage  # QPixmap,

import numpy as np
from numpy.linalg import norm

from PyQt5.QtWidgets import *
from functools import partial

import sys


class windCirc(QWidget):
    def __init__(self, par):
        super().__init__()
        self.rad = 50
        self.par = par
        self.mag_fact = 5
        self.w_influence = 0

        self.v = np.zeros(2)
        self.user = False
        self.setFixedSize(self.rad, self.rad)

        # todo add lables,draw lines, on lables have val
        # todo dif color for line circ, point for center, curser
        # todo numpy? todo check, arctan, scale todo mouse press, drag
        #
        self.pos = np.zeros(2)  # (0,0)

    def new_w(self):
        ang = np.random.random() * np.pi * 2
        mag = np.random.random() * self.mag_fact

        unit_vect = np.array((np.cos(ang), np.sin(ang)))
        self.v = unit_vect * mag
        self.pos = self.v * self.rad / self.mag_fact + np.ones(2) * self.rad / 2

        self.par.cen.set_wind(self.v * self.w_influence)
        self.update()

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton and self.user:
            # print('read press')
            pos = np.array((event.x(), event.y()))

            pos_v = pos - np.ones(2) * self.rad / 2
            dist = norm(pos_v)
            if dist <= self.rad:
                # scaled_v =
                self.v = pos_v * self.mag_fact / self.rad

                self.pos = pos
                self.par.cen.set_wind(self.v * self.w_influence)

            self.update()

    def paintEvent(self, event):

        rad = 2
        painter = QPainter(self)
        painter.eraseRect(self.rect())
        painter.setPen(Qt.black)
        painter.drawEllipse(*self.pos, rad, rad)
        painter.drawEllipse(self.rect().center(), rad, rad)
        painter.drawEllipse(self.rect())


class wind(QWidget):
    # noinspection PyArgumentList
    def __init__(self, par):
        super().__init__()
        self.layout = QVBoxLayout()
        self.check = QCheckBox('User defined')
        self.timer = QTimer()
        self.setLayout(self.layout)

        self.w = windCirc(par)
        self.timer.timeout.connect(self.w.new_w)

        self.layout.addWidget(self.w)
        self.layout.addWidget(self.check)
        self.w_influence = QSlider()
        self.w_influence.setRange(0, 100)
        self.layout.addWidget(self.w_influence)
        self.w_influence.valueChanged.connect(self.v_ch)
        self.check.clicked.connect(self.set_user)
        self.timer.setInterval(2000)
        self.timer.start()

    def v_ch(self):
        self.w.w_in = self.w_influence.value() / 100

    def set_user(self):

        if self.wv.isChecked():

            if self.check.isChecked():

                self.w.setStyleSheet('background-color: blue};')
                self.w.user = True
                self.timer.stop()
            else:

                self.w.setStyleSheet('background-color: white};')
                self.w.user = False
                self.timer.start()
        else:

            self.w.setStyleSheet('background-color: white};')
            self.w.user = False
            self.timer.stop()
            self.w.pos = self.w.rad / 2


class bird:  # (QWidget):
    def __init__(self, par):
        self.add_late = []
        self.sitting = QTimer()
        self.scale_ob = 2
        self.taken_off = QTimer()
        self.sitting.setSingleShot(True)
        self.taken_off.setSingleShot(True)
        self.taken_off.setInterval(1000)
        self.point_ls = []
        # super().__init__()
        self.par = par
        self.scale_factor = 0.2
        self.min_z = -1000

        self.sitting.timeout.connect(self.lift_off)
        self.pos = np.zeros(self.par.dimension)
        self.view_ang = 145  # deg from cen,
        self.view_dist = 50
        self.crowd_dist = 10

        self.v = np.zeros(self.par.dimension)
        self.a = np.zeros(self.par.dimension)
        # self.image = QImage(QSize(*self.dims), QImage.Format_RGB32)

        self.max_a = 2
        self.max_v = 5
        self.relative_weights = self.par.vals['Relative Weights']
        self.rel_angle = 0
        self.r1, self.r2 = 0, 0
        self.gen_rect()

    def norm_n(self, new_v):
        new_norm = norm(new_v)
        if new_norm == 0:
            unit_v = np.zeros(self.par.dimension)
        else:
            unit_v = new_v / new_norm
        return unit_v

    def gen_rect(self):
        self.rel_angle = np.abs(np.arctan(self.par.dims[1] / (self.par.dims[0] / 3)))
        self.r1 = self.par.dims[0] * 2 // 3
        self.r2 = self.par.dims[1] / np.cos(self.rel_angle)

    def rot_rect(self, painter):
        def np_r(a):
            return np.array((np.cos(a), np.sin(a)))

        angle = np.arctan2(self.v[1], self.v[0])
        points = [np_r(angle) * self.r1 + self.pos,
                  np_r(angle + np.pi - self.rel_angle) * self.r2 + self.pos,
                  np_r(angle + np.pi + self.rel_angle) * self.r2 + self.pos]

        for i in range(3):
            painter.drawLine(*points[i], *points[(i + 1) % 3])
        pass

    def move(self):
        if not self.interact_pred():
            self.interact_bird()
        self.interact_ground()
        if not self.sitting.isActive():
            self.pos += self.v
            self.v += self.a
            v_mag = norm(self.v)

            self.v = np.min((v_mag, self.par.vals['Max V'])) * self.v / v_mag + self.par.w_v
            self.a = np.zeros(self.par.dimension)
            self.interact_boundary()
            self.draw_path()

    def interact_bird(self):
        # steering_pred = self.interact_pred()
        steer_co = self.separation(True)
        steer_sep = self.separation()
        steer_align = self.alignment()
        steer_obj = self.interact_object()
        weights = np.concatenate((self.relative_weights, [self.scale_ob]))
        weights /= norm(weights)

        for n, i in enumerate([steer_co, steer_sep, steer_align, steer_obj]):
            self.a += i * weights[n]

    def interact_ground(self):
        # land n degrees, add perameters, ground, pred wieght pred dist, ground land angle, falc perameters, type
        # if land, falc effective,

        if np.abs(self.pos[1] - self.par.boundary[1]) < self.par.vals['Ground Height']:
            n = 0
            tk = 0
            for b in self.par.flock:
                # if in view
                if b != self:
                    if b.sitting.isActive():
                        n += 1
                    if b.taken_off.isActive():
                        tk += 1

            if self.sitting.isActive():
                prob_seat = 0.1 * tk
                if np.random.rand() < prob_seat:
                    self.lift_off()

            elif not self.taken_off.isActive():
                prob_seat = 0.2 + 0.01 * n - 0.1 * tk
                if np.random.rand() < prob_seat:
                    self.land()

    def land(self):
        self.pos[1] = self.par.boundary[1] - 10
        # print('land')
        v = np.random.randint(2, 8) * 500
        self.sitting.start(v)  # timeout

    def lift_off(self):
        # print('taken off')
        self.sitting.stop()
        self.taken_off.start()
        self.start_anim(False)

    def interact_pred(self):
        pred = False
        if self.par.falc:
            dis = self.par.vals['Cohesive Dist']

            rel = self.pos - self.par.falc.pos
            ang = self.ang_rel(rel)
            dist = norm(rel)  # dis = self.vals['Cohesive Dist']  # todo user val# todo norm
            if dist <= dis and ang <= self.par.vals['View Angle']:

                unit_v = self.norm_n(rel)
                rel_v = self.par.vals['Max V'] * unit_v
                v = np.min((self.par.vals['Max A'], norm(rel_v - self.v)))
                if self.par.binary['scatter']:
                    self.relative_weights[1] = 0
                else:
                    self.a = v * unit_v
                    pred = True
        return pred

    def draw_path(self):
        self.point_ls.insert(0, (self.pos[0], self.pos[1]))
        # print('point ls: ', self.point_ls)
        if len(self.point_ls) > 50:
            self.point_ls = self.point_ls[:50]

    def kill(self):
        print('kill')
        self.par.flock.remove(self)

    def interact_object(self):
        new_v = np.zeros(self.par.dimension)
        rel_m = 0

        for b in self.par.object:
            # print('object')
            rel = self.pos - b.pos
            ang = self.ang_rel(rel)
            dist = norm(rel) - b.rad
            # print('object norm')
            if dist <= 0:
                # print('object kill')
                self.kill()

            elif dist <= self.par.vals['Cohesive Dist'] and ang <= self.par.vals['View Angle']:
                rel_dis = 1 - dist / self.par.vals['Cohesive Dist']
                if rel_dis > rel_m:
                    rel_m = rel_dis

                # print('object adjust')
                new_v += rel

        unit_v = self.norm_n(new_v)

        rel_v = self.par.vals['Max V'] * unit_v
        v = np.min((self.par.vals['Max A'], norm(rel_v-self.v)))  # todo maybe add rel, where dv?
        # self.a = self.a/norm(self.a)
        return v * unit_v

    def interact_boundary(self):
        bound_3 = 100
        # loop
        for i in range(self.par.dimension):
            v = np.abs(self.v[i])
            if i == 3:
                k = bound_3
            else:
                k = self.par.boundary[i]
            if self.pos[i] <= 0:
                self.v[i] = v
            elif self.pos[i] >= k:
                self.v[i] = -v

    def ang_rel(self, p_b):
        norm_val = norm(self.v) * norm(p_b)
        if norm_val == 0:
            co_th = 0
        else:
            co_th = np.dot(self.v, p_b) / norm_val
        return np.rad2deg(np.arccos(co_th))

    def separation(self, cohesion=False):
        """not to close to nearest birds find average relative pops vector go othger way
        steer at dir with max spped unless dv/dt >>n"""

        new_v = np.zeros(self.par.dimension)

        if cohesion:
            direct = 1
            dis = self.par.vals['Cohesive Dist']
        else:
            direct = -1
            dis = self.par.vals['Crowd Dist']
        for b in self.par.flock:
            rel = b.pos - self.pos
            ang = self.ang_rel(rel)
            dist = norm(rel)
            if dist <= dis and ang <= self.par.vals['View Angle']:
                new_v += rel * direct

        unit_v = self.norm_n(new_v)

        rel_v = self.par.vals['Max V'] * unit_v
        v = np.min((self.par.vals['Max A'], norm(rel_v-self.v)))

        return v * unit_v

    def alignment(self):
        """cohesion but with v not P"""
        new_v = np.zeros(self.par.dimension)

        for b in self.par.flock:
            if b != self:
                rel = b.pos - self.pos
                ang = self.ang_rel(rel)
                dist = norm(rel)
                if dist <= self.view_dist and ang <= self.par.vals['View Angle']:
                    new_v += b.v
        unit_v = self.norm_n(new_v)

        rel_v = self.par.vals['Max V'] * unit_v
        v = np.min((self.par.vals['Max A'], norm(rel_v-self.v)))

        """usewr: max v, a, effective dist all, view angle, ammount, relivie wieghts"""
        return v * unit_v

    def start_anim(self, st=True, po=True):
        v = np.random.random(self.par.dimension)
        self.v = v / norm(v) * self.par.vals['Max V']
        if not st:
            self.v[1] = -np.abs(self.v[1])
        elif po:
            self.pos = self.par.boundary // 2
        self.pos += self.v
        self.a = np.zeros(self.par.dimension)

    def scale_z(self):
        if self.pos.size == 3:
            return self.par.vals['Size'] * 2 * (1 - (1 - self.scale_factor) / self.par.minz * self.pos[2])
        else:
            return self.par.vals['Size']


class falcon(bird):
    def __init__(self, par, pos):
        super().__init__(par)
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.finder = QTimer()
        self.timer.setSingleShot(True)
        self.finder.setSingleShot(True)
        self.sent = False
        self.timer.timeout.connect(self.change_follow)
        self.d_second = np.max(self.par.boundary)
        self.pos = np.array(pos)
        self.view_ang = 160  # deg from cen,
        self.view_dist = 100

        self.prime_tar = None
        self.second_tar = None

        self.prime_tar_dist = None

        self.close = False
        self.bc = None

    def com(self):
        pass

    def closest(self):
        d_0 = np.max(self.par.boundary)
        b_c = None
        new_v = np.zeros(self.par.dimension)
        pos = np.zeros(self.par.dimension)

        for b in self.par.flock:
            rel = b.pos - self.pos
            ang = self.ang_rel(rel)
            dist = norm(rel)
            if dist <= 0:
                b.kill()
            elif dist <= self.view_dist and ang <= self.view_ang:
                if dist < d_0:
                    d_0 = dist
                    b_c = b
        if b_c:
            new_v = b_c.v
            pos = b_c.pos

        return new_v, pos

    def closest_t(self):
        """d_c = cur dist toactive target if target is not closest: continue if time is active ie became closest 
        again, reset, if target is not closest at time, if timer not runningf start timer ewlse update secondary bird 
        at countdown end swap primary to secondary and restart """
        new_v = np.zeros(self.par.dimension)
        pos = np.zeros(self.par.dimension)
        # if prime
        for b in self.par.flock:

            # if dist(prime_tar)
            rel = b.pos - self.pos
            ang = self.ang_rel(rel)
            dist = norm(rel)
            if b == self.prime_tar:
                prime_tar_dist = dist
                if self.timer.isActive() and prime_tar_dist < self.d_second:
                    self.timer.stop()

            if dist <= 0:
                b.kill()
            elif dist <= self.view_dist and ang <= self.view_ang:
                if not self.prime_tar:
                    self.prime_tar = b
                elif dist < self.prime_tar_dist:
                    self.second_tar = b
                    if not self.timer.isActive():
                        self.timer.start()

            elif b == self.prime_tar:
                self.change_follow()
            elif b == self.second_tar:
                self.second_tar = None

        if self.prime_tar:
            new_v = self.prime_tar.v
            pos = self.prime_tar.pos
        return new_v, pos

    def change_follow(self):
        self.timer.start()
        self.prime_tar = self.second_tar

    def group_follow(self):
        new_v = np.zeros(self.par.dimension)
        pos = np.zeros(self.par.dimension)
        n = 0
        for b in self.par.flock:
            # if dist(prime_tar)
            rel = b.pos - self.pos
            ang = self.ang_rel(rel)
            dist = norm(rel)
            if dist <= 0:
                b.kill()
            elif dist <= self.view_dist and ang <= self.view_ang:

                n += 1
                pos += b.pos
                new_v += b.v
        if n > 0:
            new_v /= n
            pos /= n
        return new_v, pos

    def interact_bird(self):
        if self.par.f_vals['Type']== 'swap':
            bv, pos = self.closest_t()
        elif self.par.f_vals['Type'] == 'single':
            bv, pos = self.closest()
        else:
            bv, pos = self.group_follow()

        if norm(bv) == 0:
            if self.sent:
                # just lost
                self.finder.start()
                self.sent = False
                unit_v = self.v / norm(self.v)
            elif self.finder.isActive():
                unit_v = self.v / norm(self.v)
            else:

                sd = 2
                unit_v = np.random.normal(self.v, (sd, sd))
                unit_v /= norm(unit_v)

        else:
            if self.finder.isActive():
                self.finder.stop()
                self.sent = True

            rel_pos = pos - self.pos
            dist = norm(rel_pos)
            dt = dist / self.par.f_vals['Max V']

            bird_pos_fut = pos + dt * bv
            rel_pos_fut = bird_pos_fut - self.pos
            unit_v = self.norm_n(rel_pos_fut)
        rel_v = self.par.f_vals['Max V'] * unit_v
        v = np.min((self.par.f_vals['Max A'], norm(rel_v)))

        """usewr: max v, a, effective dist all, view angle, ammount, relivie wieghts"""
        self.v = v * unit_v

    def move(self):

        self.interact_bird()
        self.pos += self.v
        self.v += self.a
        v_mag = norm(self.v)

        self.v = np.min((v_mag, self.par.f_vals['Max V'])) * self.v / v_mag
        self.a = np.zeros(self.par.dimension)
        self.interact_boundary()
        self.point_ls.insert(0, self.pos)
        self.point_ls = self.point_ls[:50]


class obsticale:
    def __init__(self, pos, rad=5):
        # self.par = par
        self.pos = pos
        self.rad = rad
        # self.image = QImage(QSize(*self.dims), QImage.Format_RGB32)


class CannyEdge(QWidget):
    def __init__(self):
        # noinspection PyArgumentList
        super().__init__()

        self.falc = None
        self.point = True
        self.boundary = np.array((self.size().width(), self.size().height()), 'float64')
        self.minz = 1000
        self.bird_cnt = 50
        self.flock = []
        self.object = []
        self.brush_size = 3
        self.vals = {'Birds': 5, 'Size': 4, 'Max V': 10, 'Max A': 2,
                     'Cohesive Dist': 20, 'Crowd Dist': 5, 'Object D': 15, 'Ground Height': 20,
                                                                           'View Angle': 145,
                     'Relative Weights': [0.182, 0.273, 0.545]}

        self.binary = {'preditor remove': 'but', 'pref follow': ['group', 'single', 'swap'],
                       'tracepath': True, '3d': False, 'scatter': False, 'Falcon': False,
                       'Obstacle': True}  # todo pred time # todo no pred, ob in 3d# todo user timer# todo user sigma?

        self.f_vals = {'Size': 4, 'Max V': 10, 'Max A': 2, 'View Angle': 145, 'Type': 'group'}
        # self.f_but = QCheckBox('Falcon')
        # self.f_but.setChecked()
        self.dimension = 2
        self.w_v = np.zeros(self.dimension)

        self.setMinimumSize(500, 500)
        self.dims = np.array((2, 1)) * self.vals['Size']
        for i in range(self.vals['Birds']):
            self.flock.append(bird(self))

        self.normal_image = QImage(self.size(), QImage.Format_RGB32)
        self.t = QTimer()

    def set_wind(self, v):
        self.w_v = np.zeros(self.dimension)
        self.w_v[0], self.w_v[-1] = v[0], v[-1]

    def flock_event(self):
        self.update()

    def resizeEvent(self, event):
        self.boundary = np.array((self.size().width(), self.size().height()), 'float64')

    def paintEvent(self, event):

        canvas_painter = QPainter(self)
        canvas_painter.eraseRect(self.rect())
        if self.point:

            for b in self.flock:

                b.move()
                canvas_painter.setPen(QPen(Qt.black, b.scale_z()))
                canvas_painter.drawPoint(*b.pos[:2])
                if self.binary['tracepath']:
                    canvas_painter.setPen(QPen(Qt.green, 2))
                    for n in range(2, len(b.point_ls)):

                        canvas_painter.drawLine(*b.point_ls[n][:2], *b.point_ls[n - 1][:2])

            for o in self.object:
                canvas_painter.setPen(QPen(Qt.red, 2))
                canvas_painter.drawEllipse(*o.pos, o.rad, o.rad)
        else:
            canvas_painter.setPen(Qt.black)
            for b in self.flock:
                b.move()
                b.rot_rect(canvas_painter)

        if self.falc:
            canvas_painter.setPen(QPen(Qt.blue, 10))
            self.falc.move()
            canvas_painter.drawPoint(*self.falc.pos)
            if self.binary['tracepath']:
                canvas_painter.setPen(QPen(Qt.blue, 1))
                for i in self.falc.point_ls[1:]:

                    canvas_painter.drawPoint(*i)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = np.array((event.x(), event.y()))
            if not self.binary['Obstacle']:
                self.falc = falcon(self, pos)
                self.falc.start_anim()
            else:
                self.object.append(obsticale(pos, 5))

    def ani(self):

        for b in self.flock:
            b.start_anim()
        self.t.timeout.connect(self.update)
        self.t.start(20)

    def reset_v(self, v, val):
        if v == 'Relative Weights':
            val2 = v  # [int(vi) for vi in val.split(',')]

        else:
            val2 = int(val)

            if v == 'Size':
                self.dims = np.array((2, 1)) * val2
                for b in self.flock:
                    b.gen_rect()
                pass
            elif v == 'Birds':
                self.vals[v] = int(val)
                self.flock = []

                for i in range(int(val)):
                    self.flock.append(bird(self))
                self.ani()

        if self.binary['Falcon'] and v in self.f_vals:
            self.f_vals[v] = val2
        else:
            self.vals[v] = val2

    def clear_falc_ob(self):
        self.object = []
        self.falc = None

    def d3(self):
        if self.binary['3d']:
            self.boundary = np.concatenate((self.boundary, [self.minz]))
            print('3d')
            self.dimension = 3
            self.obsicalal_en(False)
        else:
            self.boundary = self.boundary[:2]
            self.dimension = 2
            self.obsicalal_en(True)
        self.set_wind(self.w_v)
        self.ani()

    def obsicalal_en(self, en):
        if not en:
            self.clear_falc_ob()
            # todo self.par.set read false
class Window(QMainWindow):
    # noinspection PyArgumentList
    def __init__(self):
        super().__init__()
        self.setWindowTitle('QMainWindow')
        self.cen = CannyEdge()
        self.setCentralWidget(self.cen)
        self._create_tools()

        self.cen.ani()

    # noinspection PyArgumentList
    def _create_tools(self):
        self.tool_dock = QDockWidget('ToolBar')
        self.tools = QWidget(self)
        self.tool_dock.setWidget(self.tools)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tool_dock)
        self.tl = QVBoxLayout()
        self.tool_layout = QGridLayout()

        self.in_p = {}
        n = 0
        m = 0

        for i, k in self.cen.vals.items():
            la = QLabel(i)
            j = QLineEdit()
            if isinstance(k, list):
                j.setReadOnly(True)
                kk = ','.join(str(iii) for iii in k)
            else:
                kk = str(k)
            j.setText(kk)
            j.editingFinished.connect(partial(self.new_vals, i))
            # sl.valueChanged.connect(partial(self.up_vect, i))

            # if i == liist
            self.in_p[i] = j

            self.tool_layout.addWidget(j, m + 1, n)
            self.tool_layout.addWidget(la, m, n)
            n += 1
            if n > 3:
                n = 0
                m += 2

        # d_0= n == 0
        if n != 0:
            n = 0
            m += 2
        for i, k in self.cen.binary.items():

            if isinstance(k, bool):
                j = QCheckBox(i)
                j.setChecked(k)
                j.clicked.connect(partial(self.set_bin, i, 0))
            elif isinstance(k, list):
                j = QComboBox()
                j.addItems(k)
                j.setCurrentText(self.cen.f_vals['Type'])
                j.currentTextChanged.connect(partial(self.set_bin, i, 1))
            else:
                j = QPushButton(i)
                j.clicked.connect(partial(self.set_bin, i, 2))
            self.tool_layout.addWidget(j, m, n)

            self.in_p[i] = j
            n += 1
            if n > 3:
                n = 0
                m += 1

        if n == 0:
            m -= 1
        self.tri = triParent(self)
        # mk = max(m - 2, 1)
        self.tool_layout.addWidget(self.tri, 0, 4, m + 1, 1)
        self.tl.addLayout(self.tool_layout)
        self.w = wind(self)
        self.tl.addWidget(self.w)
        self.tools.setLayout(self.tl)

    def set_bin(self, i, x):
        if x == 0:
            self.cen.binary[i] = self.in_p[i].isChecked()
            if i == '3d':
                self.cen.d3()
        elif x == 1:
            # todo change _list
            self.cen.f_vals['Type'] = self.in_p[i].currentText()
        else:
            self.cen.clear_falc_ob()

    #
    #     self.op = {'Scatter': False, 'Kill falc': 'but', 'swap time'}

    # def swap_b(self, b):
    #     if falc.select:
    #         grey = ['Birds', 'Dimension', 'Cohesive Dist', 'Crowd Dist', 'Relative Weights']
    #     else:
    #         ungrey all
    #     pass
    #     or hame all

    def new_vals(self, v):
        val = self.in_p[v].text()
        self.cen.reset_v(v, val)

    def rel_v_set(self, v):
        self.in_p['Relative Weights'].setText(','.join(str(iii) for iii in v))
        self.cen.reset_v('Relative Weights', v)


class triParent(QWidget):
    # noinspection PyArgumentList
    def __init__(self, par):
        super().__init__()
        self.tri = triWidget(par)
        # mk = max(m - 2, 1)
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel('Alignment'), 0, 1, 1, 2)
        self.layout.addWidget(QLabel('Cohesion'), 2, 0, 1, 2)
        self.layout.addWidget(QLabel('Seperation'), 2, 2, 1, 2)
        self.layout.addWidget(self.tri, 1, 1, 1, 2)
        self.setLayout(self.layout)


class triWidget(QWidget):
    def __init__(self, par):
        # noinspection PyArgumentList
        super().__init__()

        self.par = par
        self.setFixedSize(50, int(50 * 0.866))
        self.w = self.width()
        self.h = int(self.w * 0.866)
        self.rel_v = np.ones(3) / 3

        self.points = np.array([(self.w // 2, 0), (0, self.h), (self.w, self.h)])
        self.norm_vect = np.array([(0, 1), (0.866, -0.5), (-0.866, -0.5)])
        self.pos = np.zeros(2)  # (0,0)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            pos = np.array((event.x(), event.y()))

            if pos[1] >= np.abs(-2 * self.h / self.w * pos[0] + self.h):

                for i in range(3):
                    vect = pos - self.points[i]

                    dot_r = np.dot(self.norm_vect[i], vect)

                    self.rel_v[i] = np.round(1 - dot_r / self.h, 3)

                # self.par.vals['Relative Weights'] = self.rel_v
                self.par.rel_v_set(self.rel_v)
                self.pos = pos
            self.update()

    def paintEvent(self, event):

        rad = 2
        painter = QPainter(self)
        painter.eraseRect(self.rect())
        painter.setPen(Qt.black)
        painter.drawEllipse(*self.pos, rad, rad)
        for i in range(3):
            painter.drawLine(*self.points[i], *self.points[(i + 1) % 3])
        # self.eq_l =

        # self.eq_l =


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
