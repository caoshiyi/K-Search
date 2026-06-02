# AppendStride

**页面ID:** atlasopapi_07_00731  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00731.html

---

#### 函数功能

向后扩展一个步长值，如果扩展的步长数量超出Stride的最大限制，那么本函数不做任何事情。

#### 函数原型

```
Stride& AppendStride(const int64_t value)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 扩展的步长值。 |

#### 返回值说明

this引用。

#### 约束说明

无

#### 调用示例

```
Stride stride0({3, 256, 256});
stride0.AppendStride(1024); // 3,256,256,1024
```
