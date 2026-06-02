# E13029 Compilation_Error_Load_Custom_Fusion_Pass

**页面ID:** atlaserrorcode_15_0345  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/atlaserrorcode_15_0345.html

---

#### Symptom

Failed to load custom fusion pass lib %s. Reason: %s.

#### Solution

Analyze the failure reason mentioned above. Below are some typical solutions for common dlopen failures:

1. Verify that the library path is correct and the file exists.
2. Ensure the library and its dependencies have the correct permissions.
3. Check that all dependencies are available using the 'ldd' command.
