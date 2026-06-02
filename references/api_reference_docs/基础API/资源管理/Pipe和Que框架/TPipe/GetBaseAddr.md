# GetBaseAddr

**页面ID:** atlasascendc_api_07_0117  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0117.html

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

根据传入的logicPos（逻辑抽象位置），获取该位置的基础地址，只在CPU调试场景下此接口生效。通常用于计算Tensor在logicPos的偏移地址即Tensor地址减去GetBaseAddr返回值。

#### 函数原型

```
inline uint8_t* GetBaseAddr(int8_t logicPos)
```

#### 参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| logicPos | 输入 | 逻辑位置类型。该类型具体说明请参考TPosition。 |

#### 约束说明

NA

#### 返回值说明

逻辑位置对应的基地址。

#### 调用示例

```
auto absAddr = GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(pos));
```
