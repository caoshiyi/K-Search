# PlatformAscendC简介<a name="ZH-CN_TOPIC_0000002554344731"></a>

在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

## 需要包含的头文件<a name="section78885814919"></a>

使用该功能需要包含"tiling/platform/platform\_ascendc.h"头文件。样例如下：

```
#include "tiling/platform/platform_ascendc.h"
```

## Public成员函数<a name="section1173524710"></a>

```
[PlatformAscendC](构造及析构函数.md)() = delete
[~PlatformAscendC](构造及析构函数.md)() = default
explicit [PlatformAscendC](构造及析构函数.md)(fe::PlatFormInfos *platformInfo): platformInfo_(platformInfo) {}
uint32_t [GetCoreNum](GetCoreNum-165.md)(void) const
SocVersion [GetSocVersion](GetSocVersion.md)(void) const
NpuArch [GetCurNpuArch](GetCurNpuArch.md)(void)const
uint32_t [GetCoreNumAic](GetCoreNumAic.md)(void) const
uint32_t [GetCoreNumAiv](GetCoreNumAiv.md)(void) const
uint32_t [GetCoreNumVector](GetCoreNumVector.md)(void) const
uint32_t [CalcTschBlockDim](CalcTschBlockDim.md)(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
void [GetCoreMemSize](GetCoreMemSize.md)(const CoreMemType &memType, uint64_t &size) const
void [GetCoreMemBw](GetCoreMemBw.md)(const CoreMemType &memType, uint64_t &bwSize) const
uint32_t [GetLibApiWorkSpaceSize](GetLibApiWorkSpaceSize.md)(void) const
uint32_t [GetResGroupBarrierWorkSpaceSize](GetResGroupBarrierWorkSpaceSize.md)(void) const
uint32_t [GetResCubeGroupWorkSpaceSize](GetResCubeGroupWorkSpaceSize.md)(void) const
```

