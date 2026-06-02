# C++标准支持

**页面ID:** atlas_ascendc_10_00060  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00060.html

---

Host侧与clang15一致，支持完整的C/C++标准。

Device侧，默认支持C++11标准，支持指定C++14、C++17、C++20。由于硬件限制，部分C++运行时能力无法支持，如：

- 不支持虚函数
- 不支持虚继承
- 不支持运行时递归
- 不支持动态malloc、new/free
- 不支持STL
- 不支持运行时typeid
- 不支持文件系统IO
- 不支持标准库下的tuple及算法类运算（相关库函数调用需要标记aicore）
