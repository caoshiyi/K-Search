# 使用第三方asan工具检测内存错误问题

**页面ID:** troubleshooting_0054  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0054.html

---

asan是一款面向C/C++语言的第三方内存错误问题检查工具，可以检测使用已释放内存、内存越界、内存泄漏等问题，gcc4.8以上版本中已包含asan工具，无需单独安装。

1. 查找到asan库的路径。

可参考以下示例命令查找：

```
find *待查找目录* -name "libasan*"
```

本节以asan库在/usr/lib64/libasan.so.6.0.0路径下为例。

2. 执行应用程序时带上asan库的路径。

可参考以下示例命令（*main*表示应用程序的可执行文件，需根据实际情况修改）：

```
LD_PRELOAD=*/usr/lib64/libasan.so.6.0.0* ./*main*
```

3. 应用程序执行时，若asan检查出了内存问题，用户可参考asan打印日志分析代码问题。

asan检测出内存泄漏的报错日志示例如下，表示使用malloc接口申请10M内存，但没有使用free接口释放：

```
ERROR: LeakSanitizer: detected memory leaks

Direct leak of 10485760 byte(s) in 1 object(s) allocated from:
    #0 0xffff98d55034 in **malloc **(/usr/lib64/libasan.so.6.0.0+0xa5034)
    #1 0xffff93de3500 in SvmInit (/usr/local/Ascend/driver/lib64/driver/libascend_hal.so+0xf3500)
    #2 0xffff9975da48  (/lib/ld-linux-aarch64.so.1+0x3a48)
    #3 0xffff9975db30  (/lib/ld-linux-aarch64.so.1+0x3b30)
    #4 0xffff98c2d190 in _dl_catch_exception (/usr/lib64/libc.so.6+0x12c190)
    #5 0xffff99765f58  (/lib/ld-linux-aarch64.so.1+0xbf58)
    #6 0xffff98c2d130 in _dl_catch_exception (/usr/lib64/libc.so.6+0x12c130)
    #7 0xffff997662c4  (/lib/ld-linux-aarch64.so.1+0xc2c4)
    #8 0xffff98b7c180  (/usr/lib64/libc.so.6+0x7b180)
    #9 0xffff98c2d130 in _dl_catch_exception (/usr/lib64/libc.so.6+0x12c130)
    #10 0xffff98c2d1fc in _dl_catch_error (/usr/lib64/libc.so.6+0x12c1fc)
    #11 0xffff98b7bc5c  (/usr/lib64/libc.so.6+0x7ac5c)
    #12 0xffff98b7c254 in dlopen (/usr/lib64/libc.so.6+0x7b254)
    #13 0xffff98cfb5d0  (/usr/lib64/libasan.so.6.0.0+0x4b5d0)
```
