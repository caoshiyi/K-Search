# SetHF32TransMode

**页面ID:** atlasascendc_api_07_0259  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0259.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置HF32模式取整的具体方式，需要先使用SetHF32Mode开启HF32取整模式。

#### 函数原型

```
__aicore__ inline void SetHF32TransMode(bool hf32TransMode)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| hf32TransMode | 输入 | Mmad HF32取整模式控制入参，bool类型。支持如下两种取值： - true：则FP32将以向零靠近的方式四舍五入为HF32。- false：则FP32将以最接近偶数的方式四舍五入为HF32。 |

#### 约束说明

无

#### 调用示例

```
bool hf32TransMode = true;
AscendC::SetHF32TransMode(hf32TransMode);
```
