from dataclasses import dataclass


TOKEN_LAYOUT_64_VERSION = "geometrize_64_v1"
TOKEN_LAYOUT_192_VERSION = "geometrize_192_v1"
TOKEN_LAYOUT_VERSION = "geometrize_256_v1"

XY_BINS_PER_PIXEL = 2
ANGLE_BINS_PER_DEGREE = 3
SIZE_BINS_PER_PIXEL = 4


@dataclass(frozen=True)
class GeometrizeTokenLayout:
    x_bins: int
    y_bins: int
    angle_bins: int
    width_bins: int
    height_bins: int
    shape_bins: int = 256
    color_bins: int = 128
    special_bins: int = 3
    version: str = "unknown"

    @property
    def x_offset(self):
        return 0

    @property
    def y_offset(self):
        return self.x_offset + self.x_bins

    @property
    def angle_offset(self):
        return self.y_offset + self.y_bins

    @property
    def width_offset(self):
        return self.angle_offset + self.angle_bins

    @property
    def height_offset(self):
        return self.width_offset + self.width_bins

    @property
    def shape_offset(self):
        return self.height_offset + self.height_bins

    @property
    def red_offset(self):
        return self.shape_offset + self.shape_bins

    @property
    def green_offset(self):
        return self.red_offset + self.color_bins

    @property
    def blue_offset(self):
        return self.green_offset + self.color_bins

    @property
    def special_offset(self):
        return self.blue_offset + self.color_bins

    @property
    def vocab_size(self):
        return self.special_offset + self.special_bins

    @property
    def pad_token(self):
        return self.special_offset

    def field_slice(self, field_name):
        offset = getattr(self, f"{field_name}_offset")
        if field_name in {"red", "green", "blue"}:
            bins = self.color_bins
        else:
            bins = getattr(self, f"{field_name}_bins")
        return slice(offset, offset + bins)

    def to_metadata(self):
        return {
            "version": self.version,
            "vocab_size": self.vocab_size,
            "bins": {
                "x": self.x_bins,
                "y": self.y_bins,
                "angle": self.angle_bins,
                "width": self.width_bins,
                "height": self.height_bins,
                "shape": self.shape_bins,
                "red": self.color_bins,
                "green": self.color_bins,
                "blue": self.color_bins,
                "special": self.special_bins,
            },
            "offsets": {
                "x": self.x_offset,
                "y": self.y_offset,
                "angle": self.angle_offset,
                "width": self.width_offset,
                "height": self.height_offset,
                "shape": self.shape_offset,
                "red": self.red_offset,
                "green": self.green_offset,
                "blue": self.blue_offset,
                "special": self.special_offset,
            },
        }


# 64x64 checkpoint 中实际使用的历史词表布局。
TOKEN_LAYOUT_64 = GeometrizeTokenLayout(
    x_bins=129,
    y_bins=129,
    angle_bins=181,
    width_bins=193,
    height_bins=193,
    version=TOKEN_LAYOUT_64_VERSION,
)

# 192x192 checkpoint 的源词表布局。宽高仍只表示 [0, 128)。
TOKEN_LAYOUT_192 = GeometrizeTokenLayout(
    x_bins=384,
    y_bins=384,
    angle_bins=270,
    width_bins=512,
    height_bins=512,
    version=TOKEN_LAYOUT_192_VERSION,
)

# 当前 256x256 布局按右开区间编码：
# 坐标 [0, 256)、角度 [0, 90)、宽高仍为 [0, 128)。
TOKEN_LAYOUT = GeometrizeTokenLayout(
    x_bins=512,
    y_bins=512,
    angle_bins=270,
    width_bins=512,
    height_bins=512,
    version=TOKEN_LAYOUT_VERSION,
)

# 兼容历史迁移工具的旧名称。
OLD_TOKEN_LAYOUT = TOKEN_LAYOUT_64


if TOKEN_LAYOUT_64.vocab_size != 1468:
    raise RuntimeError("64x64 历史词表布局计算错误")
if TOKEN_LAYOUT_192.vocab_size != 2705:
    raise RuntimeError("192x192 词表布局计算错误")
if TOKEN_LAYOUT.vocab_size != 2961:
    raise RuntimeError("256x256 词表布局计算错误")
