# IrInstanceNum

**页面ID:** atlasascendc_api_07_1016  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1016.html

---

#### 功能说明

基于算子的IR定义，声明实例化时每种输入的实际个数。

#### 函数原型

```
ContextBuilder &IrInstanceNum(std::vector<uint32_t> instanceNum)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| instanceNum | 输入 | 基于算子IR原型定义，按照输入的index顺序声明实例化个数。 |

#### 返回值说明

当前ContextBuilder的对象

#### 约束说明

必须配合NodeIoNum一同使用。

#### 调用示例

```
context_ascendc::ContextBuilder builder;
(void)builder.NodeIoNum(5,3)   // 算子有5种输入，3种输出
             .IrInstanceNum({1, 1, 3, 1, 1});  // 算子实例化时，index为2的动态类型输入tensor有3个实例
```
