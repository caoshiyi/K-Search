# msaicerr工具使用指导

**页面ID:** troubleshooting_0099  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0099.html

---

msaicerr工具可用于分析AI Core Error问题、解析Dump文件、检查环境。

#### 使用约束

1. 该工具仅支持**本地分析使用**，即部署该工具的环境应该和日志所在环境为同一环境（运行环境）。
2. 该工具依赖**python3.7.5或以上版本**，在安装该工具的环境中需提前安装python。
3. 该工具**不支持**在Ascend RC形态下使用。
4. 该工具暂不支持分析以下算子的AI Core Error问题：

  - MatmulAllReduce类算子
  - MatmulAllReduceAddRmsNorm
  - MatmulAllReduceInplaceAddRmsNorm
  - AllGatherMatmul
  - MatmulReduceScatter
  - GroupedMatmulAllReduce
  - MemSet
  - NonMaxSuppressionBucketize

#### 前提条件

已在CANN运行环境上安装Toolkit软件包。详细安装请参见《CANN 软件安装指南》。

安装CANN软件后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行**source $*{install_path}*/set_env.sh**命令设置环境变量。其中${install_path}为CANN软件的安装目录，例如：/usr/local/Ascend/cann。

使用msaicerr工具前，需先进入到msaicerr.py脚本所在的目录“${install_path}/tools/msaicerr/msaicerr”。

#### 分析AI Core Error问题

- **功能说明**

分析AI Core Error问题的故障信息，辅助定位AI Core Error问题。

执行业务时，若日志文件或屏幕打印信息中包含如下AI Core Error报错，此时，需要先获取AI Core Error问题相关的故障信息（可参见收集AI Core Error问题信息），再配合使用msaicerr分析AI Core Error问题的故障信息，辅助定位AI Core Error问题。

```
# 报错示例
there is an xx aicore error

# 或报错示例
there is an xx aivec error
```

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **注意事项**

在收集到的故障信息中，请提前检查dfx/data-dump目录下是否存在dump文件、是否存在异常算子编译信息（算子编译*.o和*.json文件），检查dfx/log/host/cann目录下是否存在日志文件，若不存在，则无法使用msaicerr工具提取AI Core Error信息。

- **命令格式**

```
**python3 msaicerr.py -p** *path1* **-out** *path2* **-dev** *0*
```

- **参数说明**

  - **-p或--report_path**：必选参数，分析AI Core Error问题时用于指定AI Core Error故障信息所在的目录。不能进入-p参数指定的目录或子目录下执行msaicerr工具，否则，会出现工具解析卡住或失败的情况。
  - **-out或--output_path**：可选参数，指定解析结果文件的存放路径，如果不指定，则解析结果默认存放在执行命令的当前路径下。-out参数指定的目录不能为-p参数指定的目录或子目录，否则，会出现工具解析卡住或失败的情况。若-out参数指定值为空或无效字符串、或指定目录无写权限、或创建目录失败，则msaicerr工具退出并报错。
  - **-dev或--device_id**：可选参数，指定运行内置算子样例的Device ID，不设置该参数时，默认Device ID为0。在分析AI Core Error问题时，msaicerr工具会运行一个内置算子样例，用于检查软硬件环境是否正常。

- **使用示例**

```
python3 msaicerr.py -p *$HOME/aic_err_info* -out *$HOME/**result*
```

- **输出说明**

执行命令后，用户根据终端界面提示的info.txt文件所在的路径，通过info.txt文件中的提示信息进行问题分析和定位，info.txt文件示例及各类问题的分析方法请参见使用msaicerr工具分析AI Core Error问题。若故障信息中存在多个AI Core Error问题，则msaicerr工具按日志时间解析第一次出现的AI Core Error问题。

在执行msaicerr.py工具后，在执行msaicerr.py工具的同级目录下，会生成“debug_info.txt”或“info_{时间戳}/debug_info.txt”文件，用于记录工具执行过程中的日志信息。

#### 解析Dump文件

- **功能说明**

将Dump文件解析成.bin或.npy文件，文件中记录算子的输入、输出、workspace等信息。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **命令格式**

```
**python3 msaicerr.py -d** *path1* **-out** *path2*
```

- **参数说明**

  - **-d或--data**：必选参数，解析Dump文件时用于指定Dump文件路径，包含文件名。
  - **-out或--output_path**：可选参数，指定解析结果文件的存放路径，如果不指定，则解析结果默认跟Dump文件存放在同一路径下。

- **使用示例**

```
python3 msaicerr.py -d */demo/extra-info/data-dump/0/**exception_info.2.1.20250611171538370*
```

- **输出说明**

```
[INFO] The dump file directory will be used to as the output directory of the parsed results.
[INFO] Parse dump file finished, result path is: /demo/dfx/data-dump/0
```

根据提示，获取解析结果文件。

在执行msaicerr.py工具后，在执行msaicerr.py工具的同级目录下，会生成debug_info.txt文件，用于记录工具执行过程中的日志信息。若debug_info.txt中提示Can not read with dtype *xxx*，则表示存在工具不能识别的数据类型，需由用户自行安装第三方库文件，例如，若提示Can not read with dtype bfloat16，则需安装bfloat16ext库。

#### 检查环境

- **功能说明**

运行内置算子样例检查软硬件环境。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **命令格式**

```
**python3 msaicerr.py -e** **-dev** *0*
```

- **参数说明**

  - **-e或--env**：必选参数，表示检查环境。
  - **-dev或--device_id**：可选参数，指定运行内置算子样例的Device ID，不设置该参数，默认Device ID为0。msaicerr工具会运行一个内置算子样例，用于检查软硬件环境是否正常。

- **使用示例**

```
python3 msaicerr.py -e
```

- **输出说明**

```
[INFO] Total device count: 1
[INFO] Valid device_id 0
[INFO] Get soc_version: xxxxxxx
[INFO] Start to test env with golden op.
[INFO] The build-in sample operator runs successfully, The environment is normal.
```

在执行msaicerr.py工具后，在执行msaicerr.py工具的同级目录下，会生成debug_info.txt文件，用于记录工具执行过程中的日志信息。
