# 算子输入args下发前后不一致

**页面ID:** troubleshooting_0017  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0017.html

---

#### 分析结果

如果在info.txt中给出如下分析结论，则一般为args错误：

```
"**********************Root cause conclusion******************"
If the arguments are inconsistent before and after operator execution, memory access may be out of bounds. You are advised to use the memory error detection model to locate the fault.
```

同时在info.txt文件中“4. Operator Input/Output Memory”处会有如下信息：

```
****************4. Operator Input/Output Memory*******************
input[0] addr: 0x124080042000 end_addr:0x124080042100 size: 0x100
input[1] addr: 0x124080022000 end_addr:0x124080022008 size: 0x8
input[2] addr: 0x0 end_addr:0x4 size: 0x4
output[0] addr: 0x0 end_addr:0x8 size: 0x8
workspace_bytes:0

args before execute: [[0x124080042000, 0x124080022000, 0x124080032000, 0x124080052000, 0x1240003e5070, 0x124080010000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x124080010000, 0x100000001, 0x100000040, 0x100000002]]
args after  execute: [[0x124080042000, 0x124080022000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x124080010000, 0x100000001, 0x100000040, 0x100000002]]
```

#### 故障根因

观察上面打印的args before execute与args after execute内容，可以发现args在下发前后不一致。args为算子kernel的入参，前几位是输入、输出、workspace、tiling_gm（表示存放tiling数据的内存）的地址，如果args错误，则有可能导致AI Core error。

#### 处理方法

出现上面情况，参考以下方法处理：

打开算子踩内存检测开关（参考下文描述），再通过asys复跑业务收集故障信息，通过[https://gitee.com/ascend网站](https://gitee.com/ascend)提交issue获取帮助。

**推理场景**：执行ATC模型转换，通过--op_debug_config调测选项使能内存检测功能。

假设使能Global Memory内存检测功能的配置文件名称为*gm_debug.cfg*，文件内容配置示例如下：

```
op_debug_config=ccec_O0,ccec_g,oom
```

将该文件上传到ATC工具所在服务器，例如上传到*$HOME/module*，使用示例如下：

```
--op_debug_config=$HOME/module/gm_debug.cfg
```

**训练场景**：通过修改NPU默认配置项npu.global_options().op_debug_config使能内存检测功能。

需要修改默认配置项，在初始化NPU设备前设置全局配置项，调用示例如下：

```
import npu_device as npu
npu.global_options().op_debug_config="/root/gm_debug.cfg"
npu.open().as_default()
```

其中，gm_debug.cfg文件信息为：

```
op_debug_config = ccec_O0,ccec_g,oom
```

> **注意:** 

- --op_debug_config调测选项的详细介绍请参考《ATC离线模型编译工具用户指南》。
- npu.global_options().op_debug_config配置项的详细介绍请参考《TensorFlow 2.6.5模型迁移指南》。
