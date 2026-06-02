# Promote

**页面ID:** atlasopapi_07_00414  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00414.html

---

#### 函数功能

Promote类用于表示输出数据类型为输入或属性指定的数据类型间的提升类型。

#### 函数原型

```
class Promote {
 public:
  Promote(const std::initializer_list<const char *> &syms);
  std::vector<const char *> Syms() const; // 返回参与类型提升的符号名
  Promote(const Promote &other) = delete;
  Promote &operator=(const Promote &other) = delete;
  Promote(Promote &&other) noexcept;
  Promote &operator=(Promote &&other) noexcept;
 private:
  std::shared_ptr<void> data_;
};
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| syms | 输入 | 参与提升的类型符号列表。 符号包括输入类型的符号或者属性指定的符号。 |
| other | 输入 | 另一个Promote对象。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
