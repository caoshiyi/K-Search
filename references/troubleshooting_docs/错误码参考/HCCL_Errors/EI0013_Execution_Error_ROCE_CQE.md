# EI0013 Execution_Error_ROCE_CQE

**页面ID:** atlaserrorcode_15_0349  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/atlaserrorcode_15_0349.html

---

#### Symptom

An error CQE occurred during operator execution. Local information: server %s, device ID %s, device IP %s. Peer information: server %s, device ID %s, device IP %s.

#### Possible Cause

1. The network between two devices is abnormal. For example, the network port is intermittently disconnected.
2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.

#### Solution

1. Check whether the network devices between the two ends are abnormal.
2. Check whether the peer process exits first. If yes, check the cause of the process exit.
