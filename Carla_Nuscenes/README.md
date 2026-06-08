# CARLA NuScenes Dataset Generator

使用CARLA模拟器生成NuScenes格式的数据集！你可以通过配置文件自定义数据集。

## 环境要求

- CARLA 模拟器 (0.9.14 或更高版本)
- Python 3.7+
- 相关Python包 (见requirements.txt)

## 快速开始

### 1. 启动CARLA服务器

首先启动CARLA服务器：

```bash
cd carla_root  # 替换为你的CARLA安装目录
make launch
```

### 2. 生成数据集

运行生成脚本：

```bash
python generate.py --mode rgb --image-size 1600 900 --count 50 --random-seed 0 --root /ENS/zjw/carla_nuscenes/test
```

## 参数说明

### 必需参数

- `--mode`: 生成模式，可选值：
  - `rgb`: 生成RGB图像数据集
  - `segmentation`: 生成语义分割数据集

### 可选参数

- `--image-size WIDTH HEIGHT`: 设置图像分辨率 (默认使用配置文件中的设置)
  - 示例: `--image-size 1600 900`
- `--count`: 生成的场景数量 (默认: 5)
  - 覆盖配置文件中的count设置
- `--random-seed`: 随机种子，用于重现结果 (默认: 0)
- `--root`: 数据集输出根目录 (默认使用配置文件中的设置)

## 配置说明

### 车辆配置 (vehicles.yaml)

修改 `configs/vehicles.yaml` 文件来控制是否添加车辆：

```yaml
# 添加车辆示例
- 
  bp_name: "vehicle.audi.etron"  # 车辆类型
  location:
    x: -70.25457000732422
    y: 27.96375846862793
    z: 0.5999999642372131
  rotation:
    yaw: -90
    pitch: 0
    roll: 0
  path:  # 可选：车辆行驶路径
    - x: -87.62303161621094
      y: 12.967159271240234
      z: 0.5999999642372131

# 如果不想添加车辆，可以删除所有内容或注释掉
# []
```

### 天气配置 (config_rgb.yaml / config_segmentation.yaml)

修改配置文件中的 `weather` 部分来控制天气参数：

```yaml
weather:
  cloudiness: 0              # 云量 (0-100, 0=晴空, 100=多云)
  precipitation: 0           # 降雨强度 (0-100, 0=无雨, 100=大雨)
  precipitation_deposits: 0  # 水坑形成 (0-100)
  wind_intensity: 0          # 风强度 (0-100)
  sun_azimuth_angle: 0       # 太阳方位角 (0-360度)
  sun_altitude_angle: 90     # 太阳高度角 (-90到90度, -90=午夜, 90=正午)
  fog_density: 0             # 雾浓度 (0-100, 仅影响RGB相机)
  fog_distance: 0            # 雾起始距离 (0-无穷)
  wetness: 0                 # 湿度强度 (0-100, 仅影响RGB相机)
  fog_falloff: 0             # 雾密度衰减
  scattering_intensity: 0    # 体积雾散射强度
  mie_scattering_scale: 0    # Mie散射尺度 (大气颗粒散射)
  rayleigh_scattering_scale: 0.0331  # Rayleigh散射尺度 (空气分子散射)
  dust_storm: 0              # 沙尘暴强度 (0-100)
```

### 车辆颜色配置 (actor.py)

修改 `carla_nuscenes/actor.py` 文件中的颜色设置来控制车辆颜色：

```python
# 在Actor类的__init__方法中
if self.blueprint.id=='vehicle.tesla.model3':
    self.blueprint.set_attribute('color', '0,0,0')  # 黑色
if self.blueprint.id=='vehicle.audi.etron':
    self.blueprint.set_attribute('color', '255,255,255')  # 白色
```

**颜色格式说明**：
- `'255,255,255'` = 白色
- `'0,0,0'` = 黑色
- `'255,0,0'` = 红色
- `'0,255,0'` = 绿色
- `'0,0,255'` = 蓝色

你可以在这里为不同车型设置不同的颜色，便于在数据集中识别特定的目标车辆。

## 输出格式

生成的数据集遵循NuScenes格式，包含：

- RGB图像或语义分割图像
- 传感器校准信息
- 车辆和行人标注
- 场景元数据

## 故障排除

### 常见问题

1. **端口绑定错误**: 确保CARLA服务器正在运行且端口2000可用
2. **内存不足**: 减少场景数量或降低图像分辨率
3. **车辆生成失败**: 检查vehicles.yaml中的位置是否有效


