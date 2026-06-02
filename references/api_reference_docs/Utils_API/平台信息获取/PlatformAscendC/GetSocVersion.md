# GetSocVersion

**页面ID:** atlasascendc_api_07_1029  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1029.html

---

#### 功能说明

获取当前硬件平台版本型号。

#### 函数原型

```
SocVersion GetSocVersion(void) const
```

#### 参数说明

无

#### 返回值

当前硬件平台版本型号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN软件安装后文件存储路径下include/tiling/platform/platform_ascendc.h头文件获取。

AI处理器的型号请通过如下方式获取：

- 针对如下产品：在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，获取**Name**信息。实际配置值为AscendName，例如**Name**取值为*xxxyy*，实际配置值为Ascend*xxxyy*。

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 200I/500 A2 推理产品

Atlas 推理系列产品

Atlas 训练系列产品

- 针对如下产品，在安装昇腾AI处理器的服务器执行**npu-smi info -t board -i ***id*** -c ***chip_id*命令进行查询，获取**Chip Name**和**NPU Name**信息，实际配置值为Chip Name_NPU Name。例如**Chip Name**取值为Ascend*xxx*，**NPU Name**取值为1234，实际配置值为Ascend*xxx**_*1234。其中：

  - id：设备id，通过**npu-smi info -l**命令查出的NPU ID即为设备id。
  - chip_id：芯片id，通过**npu-smi info -m**命令查出的Chip ID即为芯片id。

Atlas A3 训练系列产品/Atlas A3 推理系列产品

#### 约束说明

无

#### 调用示例

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    // 根据所获得的版本型号自行设计Tiling策略
    // ASCENDXXX请替换为实际的版本型号
    if (socVersion == platform_ascendc::SocVersion::ASCENDXXX) {
        // ...
    }
    return ret;
}
```
