# GetScheduleMode

**页面ID:** atlasopapi_07_00249  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00249.html

---

#### 函数功能

获取算子在NPU上执行时的调度模式。

#### 函数原型

**uint32_t****GetScheduleMode****() ****const**

#### 参数说明

无。

#### 返回值说明

返回schedule_mode。

- 0：普通模式，默认情况下为普通模式。
- 1：batchmode模式，仅在模型执行场景且模型中存在核间同步算子时需要设置该模式。该模式不支持直接或者间接通过单算子API执行的方式进行算子调用的场景 。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus TilingForAdd(TilingContext *context) {
  auto schedule_mode = context->GetScheduleMode(); 
  ...
}
```
