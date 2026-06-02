# GetInstanceStart

**页面ID:** atlasopapi_07_00007  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00007.html

---

#### 函数功能

获取算子某个IR输入在实际输入中的起始序号（index）。

#### 函数原型

```
size_t GetInstanceStart() const
```

#### 参数说明

无。

#### 返回值说明

算子某个IR输入在实际输入中的起始序号（index）。

#### 约束说明

无。

#### 调用示例

```
AnchorInstanceInfo anchor_0(0, 10); //IR原型定义的第1个输入是动态输入，动态输入的实际输入个数为10
AnchorInstanceInfo anchor_1(10, 1); //IR原型定义的第2个输入是必选输入，必选输入的实际输入个数必须为1，该输入排在实际输入的第10个
auto start_of_anchor_1 = anchor_1.GetInstanceStart(); // start_of_anchor_1 = 10
```
