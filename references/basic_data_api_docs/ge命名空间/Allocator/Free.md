# Free

**页面ID:** atlasopapi_07_00280  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00280.html

---

#### 函数功能

根据指定的MemBlock释放内存到内存池。

#### 函数原型

```
virtual void Free(MemBlock *block) = 0
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| block | 输入 | 内存block指针。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

纯虚函数用户必须实现。
