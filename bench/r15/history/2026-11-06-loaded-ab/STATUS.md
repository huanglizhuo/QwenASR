# Interrupted: not a performance acceptance result

The user deferred idle-machine verification. The run was stopped by sending
SIGTERM to its own ASR child; exit -15 is intentional, not an ASR crash.
Other projects launched vitest/workerd/Dart jobs during the run. BTLEServer
also consumed approximately one CPU core. Do not treat this incomplete,
confounded run as evidence to retain or revert any optimization.

Completed 2-minute outputs and both 10-minute warmup outputs are byte-identical.
The baseline 10-minute decode varied from 33,537 ms in warmup to 58,420 ms in
the first measured run (1,765 decode tokens each). This demonstrates unstable
conditions; it does not establish the cause of the earlier reported regression.
