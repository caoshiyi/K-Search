# MemBlock构造函数和析构函数

**页面ID:** atlasopapi_07_00322  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00322.html

---

#### 函数功能

MemBlock构造函数和析构函数。

#### 函数原型

```
MemBlock(Allocator &allocator, void *addr, size_t block_size)
: allocator_(allocator), addr_(addr), count_(1U), block_size_(block_size) {}
virtual ~MemBlock() = default
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| allocator | 输入 | 用户根据Allocator派生的类的引用。 |
| addr | 输入 | device内存地址。 |
| block_size | 输入 | device内存addr的大小。 |

#### 返回值

MemBlock构造函数返回MemBlock类型的对象。

#### 异常处理

无。

#### 约束说明

用户继承Allocator后，申请内存需要返回MemBlock类型指针，用户只需按构造函数构造MemBlock对象即可，析构函数根据用户需求可以自定义，避免内存泄露。
