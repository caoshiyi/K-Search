# 收集AI Core Error问题信息

**页面ID:** troubleshooting_0006  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0006.html

---

#### 收集信息类别介绍

**注意**：在收集日志时，需收集本次AI Core Error问题附近时间的日志，旧日志中的不相关信息或其它报错信息可能导致后续msaicerr工具分析AI Core Error问题时报错。

| 信息类别 | 用途说明 |
| --- | --- |
| 应用类日志 | 在Host或Device上运行应用程序产生的调试或运行日志，用于查看用户态日志。 |
| trace日志 | 查看软件栈的维测信息。 |
| 算子exception dump文件 | dump出来的数据包括算子输入、算子数据、workspace数据等，用于后续由msaicerr工具分析问题时构造单算子用例。 |
| 异常算子编译信息 | 算子.o和.json文件，用于后续由msaicerr工具分析问题时构造单算子用例。 |
| Device侧系统类日志和其他维测信息 | 包括slog日志、syslog日志、黑匣子等，用于查看Device系统运行信息、驱动内核态信息等。 |

本节在收集AI Core Error信息时，涉及各类日志的获取、环境变量的设置，日志获取路径以默认路径来说明、环境变量设置方法以示例来说明：

- 关于日志级别、日志路径以及日志文件的详细介绍请参见《日志参考》。
- 关于环境变量的详细说明及约束，请参见《环境变量参考》。

#### 收集信息前先判断是否需要复跑业务

**注意**：如果日志中的时间与发生AI Core Error问题的时间相距较远，则旧日志可能已经被覆盖或清理，这种场景需复跑业务后，再收集故障信息。

| 场景 | 是否复跑 |
| --- | --- |
| 依赖PyTorch/MindSpore/TensorFlow框架的业务，例如**训练、在线推理**等 | **无需****复跑业务**          系统默认记录定位AI Core Error问题的信息。 |
| 不依赖或不使用PyTorch/MindSpore/TensorFlow框架的业务，例如**离线推理、单算子调用、构图**等。 | **需****复跑业务**          用户手动收集部分信息时，需要先设置环境变量、再跑业务，才能收集。例如收集算子exception dump文件时，需要先设置NPU_COLLECT_PATH环境变量。关于NPU_COLLECT_PATH的详细描述请参见《环境变量参考》中的“故障信息收集 > NPU_COLLECT_PATH”。关于手动收集信息的详细描述请参见用户手动收集步骤。          若设置了环境变量NPU_COLLECT_PATH，会导致业务变慢，影响性能，不建议长期开启，定位问题之后建议及时关闭。          > **注意:**             说明：                        性能敏感、但磁盘资源充足的情况下，后续希望在不复跑业务的情况下自动生成exception dump文件分析AI Core Error问题，可调用aclInit接口开启异常算子Dump配置（即在json文件中配置dump_scene参数）。该方式下，大部分网络，性能影响在1%以下，但开启后，多次AI Core Error问题生成的dump文件会消耗磁盘空间，用户需自行清理历史dump数据。详细接口说明请参见“aclInit”章节。 |

#### 收集信息方式介绍

定位AI Core Error问题时，需提前收集故障信息，支持以下两种方式：

| 收集方式 | 使用说明 |
| --- | --- |
| 用户手动收集 | **仅收集与AI Core Error问题有关的信息**，包括exception dump文件、算子编译信息、Host应用类日志文件。具体收集方法请参见用户手动收集步骤。 |
| 工具自动收集 | **使用asys工具，收集所有故障相关信息**（比手动收集的信息要多），包括安装版本信息、Device健康状态信息、exception dump文件、算子编译信息、全量日志文件等。具体收集方法请参见工具自动收集步骤。          **注意**：asys工具使用场景有限，集群、容器、虚拟机、云场景不支持asys工具一键式收集故障信息。 |

#### 用户手动收集步骤

- **无需复跑业务场景**

  1. 创建一个空目录（此处以aic_err_info为例），用于统一存放收集到的AI Core Error问题信息。

```
mkdir ${HOME}/aic_err_info
```

  2. 收集应用类日志、trace日志、exception dump文件、异常算子编译信息，移至${HOME}/aic_err_info目录下统一管理。

**先从以下环境变量指定的目录下获取这部分信息****：**

    - 环境变量ASCEND_PROCESS_LOG_PATH指定的路径中存放应用类日志。关于ASCEND_PROCESS_LOG_PATH的详细描述请参见“日志 > ASCEND_PROCESS_LOG_PATH”。
    - 环境变量ASCEND_WORK_PATH指定的路径中存放应用类日志、trace日志、exception dump文件。关于ASCEND_WORK_PATH的详细描述请参见“ASCEND_WORK_PATH”。
    - 若两个环境变量同时存在，则从ASCEND_PROCESS_LOG_PATH环境变量指定的路径中获取应用类日志，从ASCEND_WORK_PATH环境变量指定的路径下获取trace日志、exception dump文件、异常算子编译信息。

**若当前环境中未配置对应的环境变量，可从各类信息的默认目录下获取：**

    - 应用类日志：默认存放在Host服务器的${HOME}/ascend/log目录。
    - trace日志：默认存放在Host服务器的$HOME/ascend/atrace/目录。
    - exception dump文件、异常算子编译信息：默认存放在执行推理应用或训练脚本的当前路径的extra-info/data-dump/{device_id}目录下，该目录下包含AI Core Error相关的输入、输出数据文件exception_info.{stream_id}.{task_id}.{时间戳}，还包括异常算子编译信息，即算子.o和.json文件。

> **注意:** 

    - 如果根据以上环境变量或默认目录，没有获取到异常算子编译信息，请参见手动收集算子编译信息（算子.o和.json文件）中的指导获取这部分信息。
    - 由于Host服务器中trace日志不会自动清理，占用磁盘空间可能比较大，如果内存空间有限，用户也可以按需拷贝$HOME/ascend/atrace/目录下的指定进程的trace日志，拷贝命令示例：cp -rf $HOME/ascend/atrace/trace_*{进程组pid}** aic_err_info/
    - 此处收集的dump文件无法通过文本工具直接查看其内容，若需查看dump文件内容，先将dump文件转换为numpy格式文件后，再通过Python查看numpy格式文件，详细转换步骤请参见《精度调试工具用户指南》中的“查看dump数据文件”章节，转换后的结果示例如下：
>             

```
# 转换前的dump文件示例
aclnnMatmul_4886226_L0.MatMulCommon.3975510.1717053072064889
# 转换后的npy文件示例
aclnnMatmul_4886226_L0.MatMulCommon.3975510.1717053072064889.input.0.npy
aclnnMatmul_4886226_L0.MatMulCommon.3975510.1717053072064889.input.1.npy
aclnnMatmul_4886226_L0.MatMulCommon.3975510.1717053072064889.output.0.npy
```

  3. 收集Device侧系统类日志和其他维测信息，包括slog日志、syslog日志、黑匣子等。
        **在Host侧以root用户运行msnpureport工具**，将这部分日志导出到Host侧，然后再移动至aic_err_info目录下：

```
# 在Host上有读写权限的目录下（例如${HOME}/ascend/report）执行msnpureport命令
msnpureport** **-f

# 移动日志文件到aic_err_info目录
mv ${HOME}/ascend/report aic_err_info/
```

  4. 将${HOME}/aic_err_info目录下的信息移动到另一个目录下，目录名上可带上时间戳，便于按时间管理每一次收集的故障信息。

```
mv ${HOME}/aic_err_info  ${HOME}/aic_err_info_*timestamp*
```

- **需复跑业务场景**

  1. 创建一个空目录（此处以aic_err_info为例），用于统一存放收集到的AI Core Error问题信息。

```
mkdir ${HOME}/aic_err_info
```

  2. 设置环境变量。
        各环境变量的设置示例如下：

```
# 在ASCEND_WORK_PATH指定的目录下默认生成plog、atrace目录，分别存放应用类日志、trace日志
export ASCEND_WORK_PATH=${HOME}/aic_err_info

# 指定存放应用类日志的目录，ASCEND_PROCESS_LOG_PATH环境变量存放应用类日志的优先级更高
export ASCEND_PROCESS_LOG_PATH=${HOME}/aic_err_info/plog

# 在NPU_COLLECT_PATH指定的目录下默认生成extra-info目录，其子目录data-dump中存放exception dump文件、异常算子编译信息
export NPU_COLLECT_PATH=${HOME}/aic_err_info/
```

关于环境变量的详细说明及约束，请参见《环境变量参考》中的以下章节：

    - “ASCEND_WORK_PATH”
    - “日志 > ASCEND_PROCESS_LOG_PATH”
    - “故障信息收集 > NPU_COLLECT_PATH”

  3. 复跑业务，即再次运行用户的业务程序，可从环境变量指定的路径下获取应用类日志、trace日志、exception dump文件、异常算子编译信息。

此处要先检查下环境变量NPU_COLLECT_PATH指定的路径下是否有算子编译信息（*.o和*.json文件），如果没有，则可先从CANN软件默认安装路径为/usr/local/Ascend/cann下查找，再从默认目录${HOME}/atc_data下查找。具体查找方法请参见手动收集算子编译信息（算子.o和.json文件）。

  4. 收集Device侧系统类日志和其他维测信息的方法与不复跑业务场景下的收集方法相同，请参见3。
  5. 将${HOME}/aic_err_info目录下的信息移动到另一个目录下，目录名上可带上时间戳，便于按时间管理。

```
mv ${HOME}/aic_err_info  ${HOME}/aic_err_info_*timestamp*
```

#### 工具自动收集步骤

asys工具的使用约束请参见asys工具功能及约束，在使用asys工具前需先安装、配置asys工具，请先参见环境准备处的前提条件说明。

- **无需复跑业务场景****，执行****asys collect****命令，直接****收集故障信息****：**

```
asys collect --output=*path*
```

output表示收集信息所存放的目录，详细参数说明及约束请参见故障信息收集。

- **需复跑业务场景，执行****asys launch****命令，****同时执行业务复跑和收集故障信息****：**

```
asys launch --task="*sh ../app_run.sh*" --output=*path*
```

task表示要复跑的任务，output表示收集信息所存放的目录，详细参数说明及约束请参见业务复跑+故障信息收集。

**注意：**离线推理场景下，若需要重新构建模型（例如通过ATC工具转换模型），需先使用asys launch命令复跑构建模型的任务，再使用重新编译的模型、使用asys launch命令复跑推理业务。另外，还需将构建模型时收集的维测信息与推理时收集的维测信息放到一个目录下，例如：$HOME/asys_output。

> **注意:** 

使用工具收集信息后，需检查dfx/data-dump目录下是否存在dump文件、异常算子编译信息（算子编译*.o和*.json文件）、检查dfx/log/host/cann目录下是否存在日志文件，若不存在，则无法使用msaicerr工具提取AI Core Error信息。
