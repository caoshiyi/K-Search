# InputReorderVector

**页面ID:** atlasopapi_07_00413  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00413.html

---

#### 函数功能

支持在算子插件中调整算子的输入参数顺序，此接口为内部使用接口，外部开发者无需关注。

#### 函数原型

```
OpRegistrationData &InputReorderVector(const std::vector<int32_t> &input_order)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| input_order | 输入 | 算子输入的调整列表， 下标表示原输入索引，下标对应的值表示调整后新的输入索引。例如：第三方框架的算子A对应的昇腾AI处理器算子为AD，原输入0为in0， 原输入1为in1，原输入2为in2，插件调用接口传入input_order = {1，0，2}，那么解析后算子AD的输入0为in1， 输入1为in0，输入2为in2。 |

#### 返回值

OpRegistrationData类的引用。

#### 异常处理

无。

#### 约束说明

无。
