# ToFile

**页面ID:** atlasascendc_api_07_00118  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00118.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明

只限于CPU调试，将LocalTensor数据Dump到文件中，用于精度调试，文件保存在执行目录。

#### 函数原型

```
int32_t ToFile(const std::string& fileName) const
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| fileName | 输入 | 保存的文件名称。 |

#### 返回值说明

返回0表示数据Dump成功，非0值表示失败。

#### 约束说明

无

#### 调用示例

参考调用示例。
