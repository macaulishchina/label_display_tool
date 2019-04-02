# -*-coding:utf-8-*-
import cv2 as cv
import numpy as np
import argparse
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont


def resolve_from_cvat(result_path):
    """从cvat工具的数据解析需要绘制的数据

    Args:
        result_path: cvat工具数据文件路径

    Returns:
        工具内部使用的数据格式[DrawData]

    """
    pass


def resolve_from_coco(result_path, type='instance'):
    """从coco数据集解析需要绘制的数据

    Args:
        result_path: coco数据集文件路径
        type: coco数据集的类型，合法值（instance、keypoint），分别针对目标检测和关键点

    Returns:
        工具内部使用的数据格式[DrawData]

    """
    pass


def resolve_from_voc(result_path):
    """从vco数据集解析需要绘制的数据

    Args:
        result_path: voc数据集文件路径

    Returns:
        工具内部使用的数据格式[DrawData]

    """
    pass


def draw_one_object(image, data, style):
    pass


def draw_images(result, in_dir, out_dir, style):
    pass


def draw_videos(result, video_path, out_dir, style):
    pass


class Painter:

    def __init__(self):
        self.canvas = None

    def draw(self, draw_data, opencv_img, new_layer=True):
        """绘制[draw_data]中的数据到的图片[opencv_img]上

        Args:
            draw_data: [data_pack]内部格式数据
            opencv_img: opencv格式的图片
        """
        if args.alpha == "off":
            self.canvas = opencv_img
        else:
            canvas_shape = list(opencv_img.shape)
            canvas_shape[-1] = 4
            self.canvas = np.zeros(canvas_shape, dtype=np.uint8)
            opencv_img = cv.cvtColor(opencv_img, cv.COLOR_RGB2RGBA)

        for view in draw_data:
            base_position = None
            if 'box' in view.keys():
                base_position = view['box'][0:2]
                self._draw_box(view['name'], view['box'])
            if 'polygon' in view.keys():
                self._draw_polygon(view['polygon'])
            if 'keypoints' in view.keys():
                self._draw_keypoints(view['keypoints'])
            if 'attributes' in view.keys():
                self._draw_attributes(view['attributes'], base_position)
        if args.alpha == "on":
            res_img = cv.addWeighted(opencv_img, 1, self.canvas, args.alpha_value, 0)
        else:
            res_img = self.canvas
        cv.imwrite('out.jpg', res_img)

    def _draw_box(self, name, box_data):
        """绘制单个矩形框到图形上

        Args:
            name: 类型名
            box_data: 矩形框数据
        """
        if box_data[-1] >= args.box_confidence:
            x1, y1 = box_data[0:2]
            x2, y2 = box_data[2:4]
            title_h = int(args.name_font_size * 1.2)
            title_w = x2 - x1
            cv.rectangle(self.canvas,
                         (x1, y1 - title_h),
                         (x1 + title_w, y1),
                         color_map[args.name_font_background], -1)
            cv.rectangle(self.canvas,
                         (x1, y1), (x2, y2),
                         color_map[args.box_color],
                         args.thickness)
            self._draw_text(name,
                            (x1, y1 - title_h - args.thickness),
                            args.name_font_size,
                            color_map[args.name_font_color])

    def _draw_polygon(self, polygon_data):
        """绘制单个多边形框到图形上

        Args:
            polygon_data: 多边形框
        """
        pass

    def _draw_keypoints(self, key_points_data):
        """绘制单人的关键点到图形上

        Args:
            key_points_data: 单人的关键点
        """
        pass

    def _draw_attributes(self, attrs_data, position):
        """绘制属性值

        Args:
            attrs_data: 属性值
            position: 属性显示开始位置
        """
        a_ks = list(attrs_data.keys())
        x, y = position
        line_height = int(1.2 * args.attr_font_size)
        for i in range(len(a_ks)):
            a_k = a_ks[i]
            a_v = attrs_data[a_k]
            self._draw_text('%s:%s' % (a_k, a_v),
                            (x + args.thickness, y + i * line_height),
                            args.attr_font_size,
                            color_map[args.attr_font_color])

    def _draw_text(self, text, pt1, size, color):
        img_PIL = Image.fromarray(cv.cvtColor(self.canvas, cv.COLOR_BGR2RGB))
        font = ImageFont.truetype('NotoSansCJK-Regular.ttc', size)
        b, g, r = color
        # text = text.decode('utf8')
        draw = ImageDraw.Draw(img_PIL)
        draw.text(pt1, text, font=font, fill=(r, g, b))
        if args.alpha == "on":
            self.canvas = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2RGBA)
        else:
            self.canvas = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)


class DrawData:
    """标注工具内部使用的数据结构

    """

    def __init__(self):
        self.data_pack = []
        self.global_id = 0

    def clear(self):
        """清空数据
        """
        self.data_pack = []
        self.global_id = 0

    def add_box(self, label_name, *box_data, id=-1):
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
                self.add_box(label_name, *box_data, -1)
        return self.global_id

    def add_polygon(self, label_name, *polygon_data, id=-1):
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
                self.add_polygon(label_name, *polygon_data, -1)
        return self.global_id

    def add_keypoints(self, key_points_data, id=-1):
        """添加关键点到[data_pack]

        Args:
            key_points_data: 一个关键点的数据，数据格式[x1,y1,x2,y2,...,x17,y17,confidence],参考coco格式
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
                self.add_keypoints(*key_points_data, -1)
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

    def display(self):
        pprint(self.data_pack)

parser = argparse.ArgumentParser(description='Label display tool v0.1')
parser.add_argument('-dataset', default='coco', choices=['coco', 'voc', 'cvat'], type=str, help='输入数据集格式，默认为coco')
parser.add_argument('-subtype', default='box', choices=['box', 'point'], type=str, help='绘制图形的样式，框或者关键点')
parser.add_argument('-box_confidence', default=0.6, type=float, help='绘制矩形框的阈值')
parser.add_argument('-polygon_confidence', default=0.6, type=float, help='绘制多边形框的阈值')
parser.add_argument('-keypoints_confidence', default=0.6, type=float, help='绘制关键点的阈值')
parser.add_argument('-thickness', default=3, type=int, help='线的宽度')
parser.add_argument('-box_color', default='yellow', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='边框颜色的选择')
parser.add_argument('-name_font_size', default=20, type=int, help='字体大小')
parser.add_argument('-name_font_color', default='green', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='字体颜色')
parser.add_argument('-name_font_background', default='white', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='字体背景颜色')
parser.add_argument('-attr_font_size', default=20, type=int, help='属性字体大小')
parser.add_argument('-attr_font_color', default='white', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='属性字体颜色')
parser.add_argument('-attr_font_background', default='white', choices=['orange', 'red', 'yellow', 'green', 'blue', 'black', 'white'], type=str, help='属性字体背景颜色')
parser.add_argument('-alpha', default='off', choices=['on', 'off'], type=str, help='是否开启透明通道')
parser.add_argument('-alpha_value', default=0.2, type=float, help='透明度')

args = parser.parse_args()

color_map = {
    "orange": (0, 140, 255),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 128, 0),
    "blue": (255, 0, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255)
}

if __name__ == "__main__":
    img = cv.imread('demo.jpg')
    dd = DrawData()
    p = Painter()
    id = dd.add_box("Person中文", 250, 250, 500, 600, 0.6)
    ex = dd.append_attribute(id, hat='false', action='jump', coat='red')
    dd.display()
    p.draw(dd.data_pack, img)
