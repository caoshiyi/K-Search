# Malloc

**页面ID:** atlasopapi_07_00279  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00279.html

---

#### 函数功能

在用户内存池中根据指定size大小申请device内存。

#### 函数原型

```
virtual MemBlock *Malloc(size_t size) = 0
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| size | 输入 | 指定需要申请内存大小。 |

#### 返回值

返回MemBlock指针。

#### 异常处理

无。

#### 约束说明

纯虚函数用户必须实现。
