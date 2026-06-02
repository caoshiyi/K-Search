# GetExecuteMode

**页面ID:** atlasopapi_07_00691  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00691.html

---

#### 函数功能

返回当前算子运行的执行模式。

#### 函数原型

```
ExecuteMode GetExecuteMode() const
```

#### 参数说明

无

#### 返回值说明

返回ExecuteMode枚举值，表示当前算子的运行模式。

```
enum class ExecuteMode {
  kStaticOffloadExecute, // 静态图模式执行
  kDynamicExecute, //动态图模式执行
  kEnd  //end
};
```

#### 约束说明

无

#### 调用示例

```
ExecuteMode  GetExecuteMode(ExeResGenerationContext *context) {
  ExecuteMode mode = *context->GetExecuteMode();
  ...
}
```
