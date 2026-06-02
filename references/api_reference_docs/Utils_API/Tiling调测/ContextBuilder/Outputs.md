# Outputs

**页面ID:** atlasascendc_api_07_1012  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1012.html

---

#### 功能说明

将void* 指针的vector设置为KernelContext的output。

#### 函数原型

```
ContextBuilder &Outputs(std::vector<void *> outputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| outputs | 输入 | 保存输出的void*指针vector |

#### 返回值说明

当前ContextBuilder的对象

#### 约束说明

无

#### 调用示例

```
PlatformInfo platformInfo;
auto contextBuilder = context_ascendc::ContextBuilder().Outputs({nullptr, reinterpret_cast<void *>(&platformInfo)});
```
