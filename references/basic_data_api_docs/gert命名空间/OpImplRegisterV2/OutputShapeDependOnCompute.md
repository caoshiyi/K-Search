# OutputShapeDependOnCompute

**页面ID:** atlasopapi_07_00127  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00127.html

---

#### 函数功能

注册shape依赖于计算得到的输出列表。某些算子，比如NonZero（统计tensor中非零值的个数），计算完成前无法得知算子输出的shape信息，算子计算完成后才能获取。该类算子在原型定义时，需要使用OutputShapeDependOnCompute接口进行标识，同时在算子核函数中将实际输出shape写入到出参中，便于框架侧基于该信息进行输出内存的管理。

#### 函数原型

```
OpImplRegisterV2 &OutputShapeDependOnCompute(std::initializer_list<int32_t> outputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| outputs | 输入 | 指定输出index列表。 |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了shape依赖于计算得到的输出列表。

#### 约束说明

- 只能用于标识算子输出。
