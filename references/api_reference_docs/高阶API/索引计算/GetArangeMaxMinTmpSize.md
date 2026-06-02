# GetArangeMaxMinTmpSize

**页面ID:** atlasascendc_api_07_0857  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0857.html

---

#### 功能说明

用于获取Arange Tiling参数：Arange接口能完成计算所需最大临时空间大小max和最小临时空间大小min。

由于Arange接口内部不需要用到临时空间，max和min均返回0。

#### 函数原型

> **注意:** 

GetArithProgressionMaxMinTmpSize接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetArangeMaxMinTmpSize接口。

```
void GetArangeMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
```

```
void GetArithProgressionMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| maxValue | 输出 | Arange接口能完成计算所需最大临时空间大小。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Arange接口能完成计算所需最小临时空间大小。 |

#### 约束说明

无

#### 调用示例

```
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetArangeMaxMinTmpSize(maxValue, minValue);
```
