# Inputs

**页面ID:** atlasascendc_api_07_1011  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1011.html

---

#### 功能说明

将void* 指针的vector设置为KernelContext的inputs

#### 函数原型

```
ContextBuilder &Inputs(std::vector<void *> inputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 保存输入的void*指针vector |

#### 返回值说明

当前ContextBuilder的对象。

#### 约束说明

无

#### 调用示例

```
PlatformInfo platformInfo;
auto contextBuilder = context_ascendc::ContextBuilder().Inputs({nullptr, reinterpret_cast<void *>(&platformInfo)});
```
