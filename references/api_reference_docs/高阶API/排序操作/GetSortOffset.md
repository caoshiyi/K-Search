# GetSortOffset

**页面ID:** atlasascendc_api_07_0845  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0845.html

---

#### 功能说明

根据元素位置，获取Sort数据中的对应偏移量（单位为字节）。

#### 函数原型

```
template <typename T>
__aicore__ inline uint32_t GetSortOffset(const uint32_t elemOffset)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |

**表2 **接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| elemOffset | 输入 | 元素的位置。 |

#### 返回值说明

Sort数据中的对应偏移量。

#### 约束说明

无

#### 调用示例

```
AscendC::GetSortOffset<half>(128);
```
