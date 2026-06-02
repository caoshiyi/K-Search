# Range构造函数

**页面ID:** atlasopapi_07_00130  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00130.html

---

#### 函数功能

Range构造函数，对应如下3种构造方法：

- 可以默认构造一个上下界为nullptr的range实例;
- 也可以构造一个通过指定上下界的range实例;
- 还可以只传入一个任意类型的指针构造一个上下界相同的range实例。

#### 函数原型

```
Range() // 默认构造函数，上下界均为空指针
Range(T *min, T* max) : min_(min), max_(max) // 用户指定上界max，下界min
explicit Range(T *same_ele) : min_(same_ele), max_(same_ele) // 上下界均为same_ele
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| min | 输入 | 下界的指针，类型为T*。 |
| max | 输入 | 上界的指针，类型为T*。 |
| same_ele | 输入 | 构造相同上下界range实例时使用，上下界均使用same_ele赋值，类型为T*。 |

#### 返回值说明

返回用户指定构造的range对象。

#### 约束说明

无。

#### 调用示例

```
// 1. 默认构造
Range<int> range1; // 上下界均为nullptr
// 2. 用户指定上下界
int min = 0;
int max = 1024;
Range<int> range2(&min, &max); // 上界为1024，下界为0
// 3. 构造上下界相同的range
Range<int> range3(&min); // 上下界均为0
```
