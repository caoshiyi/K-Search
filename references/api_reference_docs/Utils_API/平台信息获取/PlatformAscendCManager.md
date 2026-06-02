# PlatformAscendCManager

**页面ID:** atlasascendc_api_07_1039  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1039.html

---

#### 功能说明

基于Kernel Launch算子工程，通过Kernel直调（Kernel Launch）方式调用算子的场景下，可能需要获取硬件平台相关信息，比如获取硬件平台的核数。PlatformAscendCManager类提供获取平台信息的功能：通过该类的GetInstance方法可以获取一个PlatformAscendC类的指针，再通过该指针获取硬件平台相关信息，支持获取的信息可参考PlatformAscendC。

> **注意:** 

- 使用该功能需要包含"tiling/platform/platform_ascendc.h"头文件，并在编译脚本中链接tiling_api、platform动态库。

  - 包含头文件的样例如下：

```
#include "tiling/platform/platform_ascendc.h"
```

  - 链接动态库的样例如下:

```
add_executable(main main.cpp)

target_link_libraries(main PRIVATE
  kernels
  tiling_api
  platform
)
```

- 当前该类仅支持如下型号：

  - Atlas 推理系列产品
  - Atlas 训练系列产品
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品

#### 函数原型

```
class PlatformAscendCManager {
public:
    static PlatformAscendC* GetInstance();
    // 在仅有CPU环境、无对应的NPU硬件环境时，需要传入customSocVersion来指定对应的AI处理器型号。注意：因为GetInstance实现属于单例模式，仅在第一次调用时传入的customSocVersion生效。
    static PlatformAscendC* GetInstance(const char *customSocVersion);
private:
...
}
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| customSocVersion | 输入 | AI处理器型号。 - 针对如下产品：在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，获取**Name**信息。实际配置值为AscendName，例如**Name**取值为*xxxyy*，实际配置值为Ascend*xxxyy*。Atlas A2 训练系列产品/Atlas A2 推理系列产品 Atlas 200I/500 A2 推理产品 Atlas 推理系列产品 Atlas 训练系列产品 - 针对如下产品，在安装昇腾AI处理器的服务器执行**npu-smi info -t board -i ***id*** -c ***chip_id*命令进行查询，获取**Chip Name**和**NPU Name**信息，实际配置值为Chip Name_NPU Name。例如**Chip Name**取值为Ascend*xxx*，**NPU Name**取值为1234，实际配置值为Ascend*xxx**_*1234。其中：  - id：设备id，通过**npu-smi info -l**命令查出的NPU ID即为设备id。  - chip_id：芯片id，通过**npu-smi info -m**命令查出的Chip ID即为芯片id。 Atlas A3 训练系列产品/Atlas A3 推理系列产品 |

#### 约束说明

无

#### 调用示例

```
GetInfoFun() {
    ...
    auto coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNum();
    ...
    return;
}
```
