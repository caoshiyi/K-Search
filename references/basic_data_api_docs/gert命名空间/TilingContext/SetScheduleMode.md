# SetScheduleMode

**页面ID:** atlasopapi_07_00248  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00248.html

---

#### 函数功能

设置算子在NPU上执行时的调度模式。

#### 函数原型

**ge****::****graphStatus********SetScheduleMode(****const********uint32_t**** s****chedule_mode****)**

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| **s****chedule_mode** | 输入 | 0：普通模式，默认情况下为普通模式。 1：batchmode模式，核间同步算子需要设置该模式。 |

#### 返回值说明

设置成功时返回“ge::GRAPH_SUCCESS”。

设置失败时返回 “ge::GRAPH_FAILED”。

关于graphStatus的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus TilingForAdd(TilingContext *context) {
  uint32_t batch_mode = 1U;
  auto ret = context->SetScheduleMode(batch_mode);  
  GE_ASSERT_SUCCESS(ret);
  ...
}
```
