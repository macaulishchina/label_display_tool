# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np
import argparse
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont
import math


class Painter:

    def __init__(self, track=False):
        self.draw_data = DrawData()
        self.line_width = args.attr_length
        self.img = None
        self.img_width = None
        self.img_height = None
        self.track = track
        self.global_color_map = {
            "orange": (0, 140, 255),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 128, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255)
        }
        self.temp_color_map = {}

    def clean(self):
        self.global_color_map = {
            "orange": (0, 140, 255),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 128, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255)
        }

    def forget(self):
        self.draw_data.clear()
        self.temp_color_map = {}

    def add_box(self, label_name, box_data, id=-1):
        return self.draw_data.add_box(label_name, box_data, id, self.track)

    def add_polygon(self, label_name, polygon_data, id=-1):
        return self.draw_data.add_polygon(label_name, polygon_data, id)

    def add_keypoints(self, key_points_data, id=-1):
        return self.draw_data.add_keypoints(key_points_data, id)

    def append_attribute(self, id, **attrs):
        return self.draw_data.append_attribute(id, **attrs)

    def add_color_map(self, names, colors):
        """添加类型到颜色的映射，没有添加的类型使用默认颜色[args.box_color]

        Args:
            names: 类型参数，例：[name_1,name_2,...,name_n]
            colors: 颜色参数，rgb格式，例：[(255,0,0), (100,255,0),...,(0,255,0)]
        """
        assert len(names) == len(colors), '类型参数和颜色参数数量不一致'
        for i in range(len(names)):
            r, g, b = colors[i]
            self.global_color_map[names[i]] = (b, g, r)

    def specify_color(self, id, color):
        """指定具体框的颜色

        Args:
            id: 框的id
            color: 颜色参数，rgb格式
        """
        b, g, r = color
        self.temp_color_map[id] = (r, g, b)

    def draw(self, img):
        """绘制[draw_data]中的数据到的图片[img]上

        Args:
            draw_data: [DrawData]对象
            img: opencv格式的图片
        """
        self.img = img
        self.img_height, self.img_width = self.img.shape[0:2]

        for view in self.draw_data.data_pack:
            if 'box' in view.keys():
                self._draw_box(view['id'], view['name'], view['box'])
            if 'polygon' in view.keys():
                self._draw_polygon(view['polygon'])
            if 'keypoints' in view.keys():
                self._draw_keypoints(view['keypoints'])
            if 'attributes' in view.keys() and 'box' in view.keys():
                self._draw_attributes(view['attributes'], view['box'])

        if self.track:
            self._draw_tracks(self.draw_data.tracks)
            self.draw_data.update_tracks()

        return self.img

    def _draw_box(self, id, name, box_data):
        """绘制单个矩形框到图形上

        Args:
            name: 类型名
            box_data: 矩形框数据
        """
        if box_data[-1] >= args.box_confidence:
            x1, y1 = box_data[0:2]
            x2, y2 = box_data[2:4]
            if self.line_width <= (x2 - x1) - 2 * args.margin:  # 大框
                label_x1 = x1 - 2 * args.thickness
                label_y1 = y1 - (args.name_font_size * 1.5 + 2 * args.thickness)
                label_x2 = label_x1 + self.line_width + 2 * args.thickness + args.margin
                label_y2 = y1
            else:  # 小框
                label_x1 = args.margin + x2 + 2 * args.thickness
                label_y1 = y1 - 2 * args.thickness
                label_x2 = label_x1 + self.line_width
                label_y2 = y1 + args.name_font_size * 1.5 - 2 * args.thickness

            # 标签背景
            layer = self.img.copy()
            cv.rectangle(layer,
                         (int(label_x1), int(label_y1)),
                         (int(label_x2), int(label_y2)),
                         self.global_color_map[args.name_font_background],
                         -1)
            self.img = cv.addWeighted(layer, args.alpha_name, self.img, 1 - args.alpha_name, 0, self.img)
            # 矩形边框
            if id in self.temp_color_map.keys():
                color = self.temp_color_map[id]
            elif name in self.global_color_map.keys():
                color = self.global_color_map[name]
            else:
                color = self.global_color_map[args.box_color]
            layer = self.img.copy()
            cv.rectangle(layer,
                         (int(x1) - args.thickness, int(y1) - args.thickness),
                         (int(x2 + args.thickness), int(y2 + args.thickness)),
                         color,
                         args.thickness * 2)
            self.img = cv.addWeighted(layer, args.alpha_box, self.img, 1 - args.alpha_box, 0, self.img)
            # 类型名称
            self._draw_text(name,
                            (int(label_x1), int(label_y1)),
                            args.name_font_size,
                            self.global_color_map[args.name_font_color])

    def _draw_polygon(self, polygon_data):
        """绘制单个多边形框到图形上

        Args:
            polygon_data: 多边形框
        """
        pass

    def _draw_keypoints(self, key_points_data):
        """绘制单人的关键点到图形上

        Args:
            key_points_data: 单人的关键点,coco格式
        """
        if args.dataset == 'coco' and key_points_data[-1] >= args.keypoints_confidence:
            part_line = {}
            # draw keypoints
            for n in range(len(key_points_data) // 3):
                if key_points_data[n * 3 + 2] <= args.keypoint_confidence:
                    continue
                cor_x, cor_y = int(key_points_data[n * 3 + 0]), int(key_points_data[n * 3 + 1])
                part_line[n] = (int(cor_x), int(cor_y))
                layer = self.img.copy()
                cv.circle(layer, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
                alpha = max(0, min(1, key_points_data[-1]))
                self.img = cv.addWeighted(layer, alpha, self.img, 1 - alpha, 0, self.img)

            # Draw limbs
            for i, (start_p, end_p) in enumerate(l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]

                    X = (start_xy[0], end_xy[0])
                    Y = (start_xy[1], end_xy[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = int(key_points_data[start_p * 3 + 2] + key_points_data[end_p * 3 + 2]) + 1
                    polygon = cv.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)

                    layer = self.img.copy()
                    cv.fillConvexPoly(layer, polygon, line_color[i])
                    # cv.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)

                    alpha = max(0, min(1, 0.5 * (key_points_data[start_p * 3 + 2] + key_points_data[end_p * 3 + 2])))
                    self.img = cv.addWeighted(layer, alpha, self.img, 1 - alpha, 0, self.img)

    def _draw_attributes(self, attrs_data, box_data):
        """绘制属性值

        Args:
            attrs_data: 属性值
            box_data: 框的信息
        """

        a_ks = list(attrs_data.keys())
        box_width = box_data[2] - box_data[0]
        line_height = args.attr_font_size * 1.5
        if self.line_width <= box_width - 2 * args.margin:  # 大框
            x, y = box_data[0], box_data[1]
        else:  # 小框
            x, y = box_data[2] + 2 * args.thickness, box_data[1] - 2 * args.thickness + args.name_font_size * 1.5
        for i in range(len(a_ks)):
            a_k = a_ks[i]
            a_v = attrs_data[a_k]
            layer = self.img.copy()
            attr_x1 = args.margin + x
            attr_y1 = y + i * line_height + (i + 1) * args.margin
            attr_x2 = args.margin + x + self.line_width
            attr_y2 = args.margin + y + (i + 1) * line_height + i * args.margin
            cv.rectangle(layer,
                         (int(attr_x1), int(attr_y1)),
                         (int(attr_x2), int(attr_y2)),
                         self.global_color_map[args.attr_font_background],
                         -1)
            self.img = cv.addWeighted(layer, args.alpha_attr, self.img, 1 - args.alpha_attr, 0, self.img)

            self._draw_text(' %s' % (a_v),
                            (attr_x1, attr_y1),
                            args.attr_font_size,
                            self.global_color_map[args.attr_font_color])

    def _draw_text(self, text, pt1, size, color):
        img_PIL = Image.fromarray(cv.cvtColor(self.img, cv.COLOR_BGR2RGB))
        font = ImageFont.truetype('NotoSansCJK-Regular.ttc', size)
        b, g, r = color
        # text = text.decode('utf8')
        draw = ImageDraw.Draw(img_PIL)
        draw.text(pt1, text, font=font, fill=(r, g, b))
        self.img = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)

    def _draw_tracks(self, tracks):
        for idx, track in enumerate(tracks):
            alpha = 1 - idx / args.track_length
            for x, y in track:
                layer = self.img.copy()
                cv.circle(layer, (int(x), int(y)), args.track_size, self.global_color_map[args.track_color], -1)
                self.img = cv.addWeighted(layer, alpha, self.img, 1 - alpha, 0, self.img)


class DrawData:
    """标注工具内部使用的数据结构

    """

    def __init__(self):
        self.data_pack = []
        self.tracks = [[]]
        self.global_id = 0

    def clear(self):
        """清空数据
        """
        self.data_pack = []
        self.global_id = 0

    def add_box(self, label_name, box_data, id=-1, track=False):
        """添加矩形框到[data_pack]

        Args:
            label_name: 框对应的标签名
            box_data: 一个矩形框的数据，格式为[x1, y1, x2, y2, confidence]
            id: 添加到[data_pack]中的绘制单位会递增添加一个id，默认id=-1会自动创建，
                指定id，如果id存在，则替代原有的绘制单位
        Returns:
            绘制单位的id

        """
        if id == -1:
            self.global_id += 1
            self.data_pack.append({
                "id": self.global_id,
                "name": label_name,
                "box": list(box_data)
            })
        else:
            for i in range(len(self.data_pack)):
                if self.data_pack[i]['id'] == id:
                    self.data_pack[i]['box'] = list(box_data)
            if i == len(self.data_pack):  # id不存在，则新建
                self.add_box(label_name, box_data, -1)

        if track:
            self.tracks[0].append(((box_data[0] + box_data[2]) / 2, (box_data[1] + box_data[3]) / 2))

        return self.global_id

    def add_polygon(self, label_name, polygon_data, id=-1):
        """添加多边形框到[data_pack]

        Args:
            label_name: 框对应的标签名
            polygon_data: 一个多边形框的数据，格式为[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence],
            id: 添加到[data_pack]中的绘制单位会递增添加一个id，默认id=-1会自动创建，
                指定id，如果id存在，则替代原有的绘制单位
        Returns:
            绘制单位的id

        """
        if id == -1:
            self.global_id += 1
            self.data_pack.append({
                "id": self.global_id,
                "name": label_name,
                "polygon": list(polygon_data)
            })
        else:
            for i in range(len(self.data_pack)):
                if self.data_pack[i]['id'] == id:
                    self.data_pack[i]['polygon'] = list(polygon_data)
            if i == len(self.data_pack):  # id不存在，则新建
                self.add_polygon(label_name, polygon_data, -1)
        return self.global_id

    def add_keypoints(self, key_points_data, id=-1):
        """添加关键点到[data_pack]

        Args:
            key_points_data: 一个关键点的数据，数据格式[x1,y1,s1,x2,y2,s2,...,x17,y17,s17,confidence],参考coco格式
            id: 添加到[data_pack]中的绘制单位会递增添加一个id，默认id=-1会自动创建，
                指定id，如果id存在，则替代原有的绘制单位
        Returns:
            绘制单位的id

        """
        if id == -1:
            self.global_id += 1
            self.data_pack.append({
                "id": self.global_id,
                "name": "person",
                "keypoints": list(key_points_data)
            })
        else:
            for i in range(len(self.data_pack)):
                if self.data_pack[i]['id'] == id:
                    self.data_pack[i]['keypoints'] = list(key_points_data)
            if i == len(self.data_pack):  # id不存在，则新建
                self.add_keypoints(key_points_data, -1)
        return self.global_id

    def append_attribute(self, id, **attrs):
        """附加属性的绘制单位上，通过id来区分绘制单位

        Args:
            id: 指定要绑定的绘制单位的id，如果id不存在则附加失败，如果id存在则添加或替换原有属性
            attrs: 附加的多属性，数据格式{'attr_name1':'attr1_value','attr_name2':'attr2_value',...}
            id: 添加到[data_pack]中的绘制单位会递增添加一个id，默认id=-1会自动创建，
                指定id，如果id存在，则替代原有的绘制单位
        Returns:
            绘制单位存在返回True，不存在返回False

        """
        for i in range(len(self.data_pack)):
            if self.data_pack[i]['id'] == id:
                if 'attributes' not in self.data_pack[i].keys():
                    self.data_pack[i]['attributes'] = attrs
                else:
                    self.data_pack[i]['attributes'].update(**attrs)
                return True
        return False

    def update_tracks(self):
        if len(self.tracks) >= args.track_length:
            self.tracks = self.tracks[:args.track_length - 1]
        self.tracks.insert(0, [])

    def display(self):
        pprint(self.data_pack)
        pprint(self.tracks)


"""
配置参数
"""
parser = argparse.ArgumentParser(description='Label display tool v0.1')
# -----------------------------------command--------------------------------------------------------
parser.add_argument('-dataset', default='coco', choices=['coco', 'voc', 'cvat', 'peta'], type=str, help='输入数据集格式，默认为coco')
parser.add_argument('-subtype', default='box', choices=['box', 'point'], type=str, help='绘制图形的样式，框或者关键点')
parser.add_argument('-box_confidence', default=0.6, type=float, help='绘制矩形框的阈值')
parser.add_argument('-polygon_confidence', default=0.6, type=float, help='绘制多边形框的阈值')
parser.add_argument('-keypoints_confidence', default=0.6, type=float, help='绘制总体关键点的阈值')
parser.add_argument('-keypoint_confidence', default=0.05, type=float, help='绘制单个关键点的阈值')
# -----------------------------------style--------------------------------------------------------
parser.add_argument('-thickness', default=2, type=int, help='线的宽度')
parser.add_argument('-margin', default=3, type=int, help='外边框间隔')
parser.add_argument('-box_color', default='green', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='边框颜色的选择')
parser.add_argument('-name_font_size', default=25, type=int, help='字体大小')
parser.add_argument('-name_font_color', default='black', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='字体颜色')
parser.add_argument('-name_font_background', default='white', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='字体背景颜色')

parser.add_argument('-attr_length', default=100, type=int, help='属性长度')
parser.add_argument('-attr_font_size', default=20, type=int, help='属性字体大小')
parser.add_argument('-attr_font_color', default='white', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='属性字体颜色')
parser.add_argument('-attr_font_background', default='black', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='属性字体背景颜色')
parser.add_argument('-alpha_box', default=0.6, type=float, help='框的透明度')
parser.add_argument('-alpha_name', default=0.4, type=float, help='名称背景透明度')
parser.add_argument('-alpha_attr', default=0.4, type=float, help='熟悉背景透明度')
parser.add_argument('-track_length', default=25, type=int, help='追踪的最大长度')
parser.add_argument('-track_color', default='orange', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='踪迹颜色')
parser.add_argument('-track_size', default=3, type=int, help='轨迹的宽度')

args = parser.parse_args()

if args.dataset == 'coco':
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
else:
    NotImplementedError

if __name__ == "__main__":
    out_dir = './demo.mp4'
    fps = 25
    img = cv.imread('demo.jpg')
    fourcc_mpeg4 = cv.VideoWriter_fourcc(*'DIVX')
    fourcc = fourcc_mpeg4
    size = (img.shape[1], img.shape[0])
    videoWrite = cv.VideoWriter(out_dir, fourcc, fps, size)

    p = Painter(track=True)

    for i in range(100):
        print(i)
        img = cv.imread('demo.jpg')
        p.forget()
        id1 = p.add_box("Person中文21", (100 + i * 20, 100, 700 + i * 20, 600, 0.6))
        id2 = p.add_box("Person中文1", [250, 250, 300, 300, 0.6])
        kps = [855.274658203125, 393.79803466796875, 0.6467858552932739, 860.8237915039062, 385.4743347167969, 0.7896369695663452, 846.9509887695312, 391.0234680175781, 0.015277005732059479, 888.5695190429688, 391.0234680175781, 0.8852718472480774, 916.3151245117188, 368.826904296875, 0.0440521202981472, 885.794921875, 463.1622619628906, 0.5772745609283447, 949.6099853515625, 393.79803466796875, 0.685591995716095,
               835.8526611328125, 502.0061950683594, 0.654134213924408, 916.3151245117188, 354.9540710449219, 0.04689516872167587, 830.3035278320312, 440.9656982421875, 0.8230804800987244, 849.7255249023438, 427.0928649902344, 0.03562654182314873, 827.5289916992188, 596.341552734375, 0.412489652633667, 877.47119140625, 590.7924194335938, 0.35453400015830994, 785.9104614257812, 654.6074829101562, 0.030688518658280373,
               852.5000610351562, 612.9889526367188, 0.011988930404186249, 902.4423217773438, 676.8040161132812, 0.01115232054144144, 902.4423217773438, 690.6768188476562, 0.018584873527288437,
               2]
        id3 = p.add_keypoints(kps)
        hat = 'true'
        ex1 = p.append_attribute(id1, hat=hat, action='jump', coat='red')
        ex2 = p.append_attribute(id2, hat='false', action='jump', coat='red')
        if hat == 'true':
            p.specify_color(ex1, (255, 0, 0))
        out = p.draw(img)
        videoWrite.write(out)
        # cv.imwrite('out.jpg', out)
        # cv.imshow('s', out)
        # cv.waitKey(0)
