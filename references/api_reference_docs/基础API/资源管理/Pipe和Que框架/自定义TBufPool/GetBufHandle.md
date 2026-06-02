# GetBufHandle

**页面ID:** atlasascendc_api_07_0131  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0131.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

根据offset取出自定义TBufPool中管理的内存块。

#### 函数原型

```
__aicore__ inline TBufHandle GetBufHandle(uint8_t offset)
```

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| offset | 输入 | 内存块的index值 |

#### 约束说明

offset值不可超出EXTERN_IMPL_BUFPOOL中定义的BUFID_SIZE大小，用户可根据需要添加校验。

#### 返回值说明

指定的内存块，类型为TBufHandle。

#### 调用示例

请参考调用示例。
