# Shape构造函数

**页面ID:** atlasopapi_07_00151  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00151.html

---

#### 函数功能

Shape构造函数。

#### 函数原型

下文中的dim_num_为维度个数，即有几维；dims_为具体的维度值信息。

- 默认构造一个shape，默认构造的shape实例中，dim_num_长度为0。

```
Shape() : dim_num_(0), dims_{0}
```

- 通过dims_值构造shape，例如：Shape({8,3,224,224})表示创建一个Shape实例，有4个维度，每个维度的值分别是8,3,224,224。

```
Shape(const std::initializer_list<int64_t> &args) : Shape()
```

- 拷贝构造，为了提升性能，dims_超过源Shape dim_num_的空间没有拷贝，可能有脏数据。

```
Shape(const Shape &other)
```

- 拷贝赋值，为了提升性能，dims_超过源Shape dim_num_的空间没有拷贝，可能有脏数据。

```
Shape &operator=(const Shape &other)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| args | 输入 | shape的所有dim值。 |
| other | 输入 | 源Shape对象。 |

#### 返回值说明

生成一个初始化的Shape对象。

#### 约束说明

无。

#### 调用示例

```
Shape shape({3, 256, 256}); // dim_num_=3  dims_的前三维的维度为3,256,256
```
