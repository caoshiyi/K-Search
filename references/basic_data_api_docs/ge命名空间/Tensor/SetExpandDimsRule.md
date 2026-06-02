# SetExpandDimsRule

**页面ID:** atlasopapi_07_00476  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00476.html

---

#### 函数功能

设置Tensor的补维规则。

补维是指在原有shape的基础上，添加一到多个维度。例如原shape[2,2]有两根轴，那么在两根轴中间补两维后的shape为[2,1,1,2]，补维后shape的第0、3根轴被称为原始轴，第1、2根轴被称为补维轴。

通过1和0描述补维规则，1代表当前轴为补维轴，0代表当前轴为原始轴，从左到右依次代表当前shape每根轴的来源，例如：

**表1 **补维规则示例

| 补维规则 | 补维前shape | 补维后shape |
| --- | --- | --- |
| 0110 | [2, 2] | [2, 1, 1, 2] |
| 100 | [2, 3] | [1, 2, 3] |
| 1000 | [2, 3] | 补维规则与补维前shape不匹配，规则指定原始轴有3根，但原始shape只有2根轴，补维报错。 |

#### 函数原型

```
graphStatus SetExpandDimsRule(const AscendString &expand_dims_rule)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| expand_dims_rule | 输入 | 待设置的expand_dims_rule补维规则，采用字符串形式表示补维。 |

#### 返回值

graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

无。
