# PlatformAscendC简介

**页面ID:** atlasascendc_api_07_00059  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00059.html

---

在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

#### 需要包含的头文件

使用该功能需要包含"tiling/platform/platform_ascendc.h"头文件。样例如下：

```
#include "tiling/platform/platform_ascendc.h"
```

#### Public成员函数

```
PlatformAscendC() = delete
~PlatformAscendC() = default
explicit PlatformAscendC(fe::PlatFormInfos *platformInfo): platformInfo_(platformInfo) {}
uint32_t GetCoreNum(void) const
SocVersion GetSocVersion(void) const
uint32_t GetCoreNumAic(void) const
uint32_t GetCoreNumAiv(void) const
uint32_t GetCoreNumVector(void) const
uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
void GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const
void GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const
uint32_t GetLibApiWorkSpaceSize(void) const
uint32_t GetResGroupBarrierWorkSpaceSize(void) const
uint32_t GetResCubeGroupWorkSpaceSize(void) const
```
