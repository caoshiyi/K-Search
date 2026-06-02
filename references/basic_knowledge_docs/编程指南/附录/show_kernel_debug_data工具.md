# show_kernel_debug_data工具

**页面ID:** atlas_ascendc_10_0102  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0102.html

---

静态图场景下，整图算子全部下沉到NPU侧执行，kernel侧单算子调试信息（通过printf接口）需要在模型执行结束后才能获取。本工具提供了离线解析能力，帮助用户获取并解析调试信息（将bin文件解析成可读格式）。

> **注意:** 

show_kernel_debug_data支持多用户并发调用，但用户需要指定不同的落盘路径，否则可能出现落盘内容被覆盖等问题。

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | x |

#### 工具安装

1. 安装工具。

工具跟随CANN软件包发布（参考环境准备完成CANN安装），其路径默认为“${INSTALL_DIR}/tools/show_kernel_debug_data”，其中${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，则安装后文件存储路径为：/usr/local/Ascend/cann。

2. 设置环境变量。

  - root用户安装Ascend-cann-toolkit包时

```
source /usr/local/Ascend/cann/set_env.sh
```

  - 非root用户安装Ascend-cann-toolkit包时

```
source ${HOME}/Ascend/cann/set_env.sh
```

3. 检查工具是否安装成功。

       执行如下命令，若能正常显示--help或-h信息，则表示工具环境正常，功能可正常使用。

```
show_kernel_debug_data -h
```

#### 使用方法

- **命令行方式**

```
show_kernel_debug_data <bin_file_path> [<output_path>]
```

| 参数 | 可选/必选 | 说明 |
| --- | --- | --- |
| <bin_file_path> | 必选 | kernel侧调试信息落盘的bin文件路径，例如“/input/dump_workspace.bin”。 |
| <output_path> | 可选 | 解析结果的保存路径，例如“/output_dir”。默认是当前命令行执行目录下。 |

- **API方式**

**表1 **show_kernel_debug_data接口说明表

| **函数原型** | def show_kernel_debug_data(bin_file_path: str, output_path: str = './') -> None |  |
| --- | --- | --- |
| **函数功能** | 获取kernel侧调试信息并解析成可读文件。 |  |
| **参数（IN）** | bin_file_path | kernel侧调试信息落盘的bin文件路径，字符串类型。 |
| output_path | 解析结果的保存路径，字符串类型，默认是当前接口调用脚本所在目录下。 |  |
| **参数（OUT）** | NA | - |
| **返回值** | NA | - |
| **使用约束** | 无 |  |
| ``` from show_kernel_debug_data import show_kernel_debug_data show_kernel_debug_data(./input/dump_workspace.bin) ``` |  |  |

#### 产物说明

工具解析结果文件目录结构如下：

```
├ ${output_path}
├── PARSER_${timestamp}           // ${timestamp}表示时间戳。
│   ├── parser.log              // 工具解析的日志，包含kernel侧日常流程和printf打印信息。
```
