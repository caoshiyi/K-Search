# NodeIoNum

**页面ID:** atlasascendc_api_07_1014  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1014.html

---

#### 功能说明

声明算子定义的输入与输出个数

#### 函数原型

```
ContextBuilder &NodeIoNum(size_t inputNum, size_t outputNum)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputNum | 输入 | 算子IR原型定义中的输入个数 |
| outputNum | 输入 | 算子IR原型定义中的输出个数 |

#### 返回值说明

当前ContextBuilder的对象

#### 约束说明

必须配合IrInstanceNum一同使用。

#### 调用示例

```
auto builder = ContextBuilder().NodeIoNum(5, 3); // 该算子有5个输入，3个输出
```
