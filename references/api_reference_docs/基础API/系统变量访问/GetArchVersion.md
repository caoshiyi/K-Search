# GetArchVersion

**页面ID:** atlasascendc_api_07_0187  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0187.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

获取当前AI处理器架构版本号。

#### 函数原型

```
__aicore__ inline void GetArchVersion(uint32_t& coreVersion)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| coreVersion | 输出 | AI处理器架构版本 数据类型：uint32_t |

#### 约束说明

在调用GetArchVersion接口前，需先定义coreVersion ，调用GetArchVersion接口后coreVersion会变成相对应架构版本号的值。

由于硬件约束，在查看转换后的AI处理器架构版本号时需要将其打印成十六进制的数或者自行转换成十六进制的数。

#### 调用示例

如下样例通过调用GetArchVersion接口获取AI处理器架构版本号。

```
uint32_t coreVersion = 0;//定义AI处理器版本
    AscendC::GetArchVersion(coreVersion);
    AscendC::PRINTF("core version is %x", coreVersion);//需用%x将其打印成十六进制的数
```

不同型号服务器有不同的架构版本号取值，如下表所示：

| 架构版本号 | 型号 |
| --- | --- |
| 200 | Atlas 推理系列产品AI Core |
| 220 | Atlas A2 训练系列产品/Atlas A2 推理系列产品 Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
