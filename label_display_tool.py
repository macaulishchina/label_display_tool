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
            for n in range(17):
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
parser.add_argument('-keypoints_confidence', default=0, type=float, help='绘制总体关键点的阈值')
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
    info = {'clerk': [[77.45024108886719,
                       287.03125,
                       0.7346054315567017,
                       68.38117218017578,
                       277.962158203125,
                       0.3233407139778137,
                       65.35814666748047,
                       287.03125,
                       0.805622398853302,
                       213.4862518310547,
                       525.8500366210938,
                       0.01372559554874897,
                       41.173973083496094,
                       314.2384338378906,
                       0.584958016872406,
                       38.15095138549805,
                       323.3074951171875,
                       0.42333802580833435,
                       65.35814666748047,
                       398.883056640625,
                       0.6346116065979004,
                       92.56535339355469,
                       338.422607421875,
                       0.36592262983322144,
                       113.72651672363281,
                       471.43560791015625,
                       0.6946117877960205,
                       143.9567413330078,
                       296.1003112792969,
                       0.3493708670139313,
                       168.1409149169922,
                       507.7118835449219,
                       0.7582465410232544,
                       168.1409149169922,
                       441.20538330078125,
                       0.36090585589408875,
                       165.11788940429688,
                       474.4586181640625,
                       0.5413099527359009,
                       204.4171905517578,
                       501.66583251953125,
                       0.31981080770492554,
                       216.50927734375,
                       537.9420776367188,
                       0.5848658680915833,
                       225.57833862304688,
                       547.0111694335938,
                       0.3079352378845215,
                       252.78555297851562,
                       577.2413940429688,
                       0.6516161561012268,
                       0.997515,
                       0.0,
                       0.508905385351712,
                       0.21697831223062963,
                       0.6406066669163081,
                       0.0,
                       0,
                       0.0,
                       0.0,
                       0.0],
                      [429.98480224609375,
                       54.88311004638672,
                       0.6922478675842285,
                       444.37591552734375,
                       40.49200439453125,
                       0.7186833024024963,
                       419.7054443359375,
                       40.49200439453125,
                       0.8478369116783142,
                       471.10223388671875,
                       17.877410888671875,
                       0.7660027742385864,
                       407.3702087402344,
                       24.045028686523438,
                       0.07003001868724823,
                       508.1079406738281,
                       73.3859634399414,
                       0.5824033617973328,
                       397.0908203125,
                       67.21834564208984,
                       0.6632924675941467,
                       545.1136474609375,
                       159.7325897216797,
                       0.46558162569999695,
                       376.5321044921875,
                       126.83863830566406,
                       0.6951289176940918,
                       481.381591796875,
                       194.6824188232422,
                       0.7974038124084473,
                       390.9232177734375,
                       167.95608520507812,
                       0.8112000823020935,
                       499.88446044921875,
                       231.68812561035156,
                       0.2986519932746887,
                       423.8171691894531,
                       227.57635498046875,
                       0.3647811710834503,
                       522.4990234375,
                       190.5706787109375,
                       0.1264675408601761,
                       380.64385986328125,
                       270.74969482421875,
                       0.011088058352470398,
                       452.599365234375,
                       256.35858154296875,
                       0.2849505841732025,
                       464.93463134765625,
                       258.41448974609375,
                       0.008363566361367702,
                       0.999307,
                       0.4526159767944477,
                       0.6733974615533693,
                       0.05795659035102312,
                       0.5653077963646619,
                       0.3830455992282981,
                       0.8887035889970956,
                       0.03574306301422647,
                       0.07089316540282048,
                       0.25969323556404]],
            'customer': [[1117.767822265625,
                          327.47760009765625,
                          0.7188237309455872,
                          1130.5677490234375,
                          327.47760009765625,
                          0.8631355166435242,
                          1124.167724609375,
                          314.6776123046875,
                          0.07215064018964767,
                          1162.56787109375,
                          353.07763671875,
                          0.6362387537956238,
                          1162.56787109375,
                          340.27764892578125,
                          0.010143641382455826,
                          1140.1678466796875,
                          461.8777160644531,
                          0.5440362691879272,
                          1172.1678466796875,
                          375.4776611328125,
                          0.2854093015193939,
                          1072.9677734375,
                          535.477783203125,
                          0.7706249356269836,
                          1092.167724609375,
                          397.877685546875,
                          0.041463062167167664,
                          1005.7677612304688,
                          487.47772216796875,
                          0.86930251121521,
                          1063.36767578125,
                          388.27764892578125,
                          0.020627988502383232,
                          1056.9677734375,
                          538.6777954101562,
                          0.3892614543437958,
                          1040.9677734375,
                          468.2777404785156,
                          0.19856974482536316,
                          986.5677490234375,
                          580.2777709960938,
                          0.8038071990013123,
                          1021.7677612304688,
                          548.2777709960938,
                          0.28843948245048523,
                          986.5677490234375,
                          637.8778076171875,
                          0.7395270466804504,
                          1002.5677490234375,
                          602.6777954101562,
                          0.3642149567604065,
                          0.999354],
                         [353.1443786621094,
                          436.4085388183594,
                          0.003669841680675745,
                          441.61968994140625,
                          494.470458984375,
                          0.004475852008908987,
                          361.4389343261719,
                          403.23028564453125,
                          0.012985304929316044,
                          267.4339294433594,
                          417.0545654296875,
                          0.2407757043838501,
                          353.1443786621094,
                          422.5842590332031,
                          0.7751919627189636,
                          237.02053833007812,
                          472.35162353515625,
                          0.6541393995285034,
                          402.9117431640625,
                          500.0001525878906,
                          0.45090627670288086,
                          253.6096649169922,
                          530.4135131835938,
                          0.1153748407959938,
                          469.2681884765625,
                          558.0620727539062,
                          0.7510033249855042,
                          361.4389343261719,
                          419.8194274902344,
                          0.007663663011044264,
                          458.20880126953125,
                          497.23529052734375,
                          0.6906856894493103,
                          333.7904052734375,
                          577.416015625,
                          0.2788110375404358,
                          411.206298828125,
                          602.2997436523438,
                          0.3799370527267456,
                          353.1443786621094,
                          629.9482421875,
                          0.1913071572780609,
                          438.85479736328125,
                          621.6536865234375,
                          0.4924483597278595,
                          267.4339294433594,
                          580.180908203125,
                          0.10104352980852127,
                          444.384521484375,
                          676.9507446289062,
                          0.6534383893013,
                          0.999293],
                         [597.9319458007812,
                          103.02485656738281,
                          0.01393226720392704,
                          605.2075805664062,
                          95.74925231933594,
                          0.04043896123766899,
                          590.6563110351562,
                          98.17445373535156,
                          0.011458409018814564,
                          631.884765625,
                          88.47364807128906,
                          0.5092360973358154,
                          622.1839599609375,
                          86.04844665527344,
                          0.006195101421326399,
                          677.963623046875,
                          112.72567749023438,
                          0.6815482974052429,
                          568.8295288085938,
                          127.27688598632812,
                          0.4061734974384308,
                          694.9400634765625,
                          195.1825714111328,
                          0.6071711778640747,
                          522.7506713867188,
                          190.3321533203125,
                          0.6076415181159973,
                          656.1367797851562,
                          263.0882263183594,
                          0.8116076588630676,
                          466.97100830078125,
                          238.83621215820312,
                          0.40464892983436584,
                          677.963623046875,
                          238.83621215820312,
                          0.3046170771121979,
                          622.1839599609375,
                          231.5605926513672,
                          0.24536705017089844,
                          660.9872436523438,
                          306.7418518066406,
                          0.03207295015454292,
                          631.884765625,
                          304.316650390625,
                          0.027881041169166565,
                          682.8140258789062,
                          265.513427734375,
                          0.045092541724443436,
                          675.5383911132812,
                          267.9386291503906,
                          0.011956053785979748,
                          0.999678],
                         [825.5571899414062,
                          663.1907348632812,
                          0.002483329037204385,
                          805.2135009765625,
                          375.4725646972656,
                          0.006311149802058935,
                          828.4634399414062,
                          439.409912109375,
                          0.005491468124091625,
                          811.0260009765625,
                          401.6287536621094,
                          0.7844724059104919,
                          895.3070678710938,
                          404.53497314453125,
                          0.42888951301574707,
                          764.5260620117188,
                          459.753662109375,
                          0.6854898929595947,
                          924.3695068359375,
                          442.316162109375,
                          0.6333245635032654,
                          749.994873046875,
                          520.7847900390625,
                          0.56544429063797,
                          889.4945678710938,
                          419.06622314453125,
                          0.1407405287027359,
                          776.1510620117188,
                          453.941162109375,
                          0.009279961697757244,
                          831.3696899414062,
                          436.503662109375,
                          0.06811719387769699,
                          784.8698120117188,
                          686.440673828125,
                          0.48887398838996887,
                          880.77587890625,
                          680.628173828125,
                          0.4879525601863861,
                          761.6198120117188,
                          637.0344848632812,
                          0.012585349380970001,
                          918.5570678710938,
                          523.6909790039062,
                          0.008048526011407375,
                          758.7135620117188,
                          671.9094848632812,
                          0.011818451806902885,
                          883.6820678710938,
                          407.44122314453125,
                          0.004517609719187021,
                          0.998848]],
            'scanner': [[919.92576, 279.054, 0.993562],
                        [397.17632000000003, 243.12708, 0.993359]]}

    for i in range(1):
        print(i)
        img = cv.imread('demo.jpg')
        p.forget()
        id1 = p.add_box("Person中文21", (100 + i * 20, 100, 700 + i * 20, 600, 0.6))
        id2 = p.add_box("Person中文1", [250, 250, 300, 300, 0.6])
        kps = info['customer'][2]
        id3 = p.add_keypoints(kps)
        hat = 'true'
        ex1 = p.append_attribute(id1, hat=hat, action='jump', coat='red')
        ex2 = p.append_attribute(id2, hat='false', action='jump', coat='red')
        if hat == 'true':
            p.specify_color(ex1, (255, 0, 0))
        out = p.draw(img)
        # videoWrite.write(out)
        cv.imwrite('out.jpg', out)
        # cv.imshow('s', out)
        # cv.waitKey(0)
