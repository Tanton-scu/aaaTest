import unittest

from prievo_agent.domain.models import Run, RunStatus
from prievo_agent.runtime.state_machine import RunStateMachine


class RunStateMachineTest(unittest.TestCase):
    def test_pause_resume_cancel_and_terminal_guard(self):
        machine = RunStateMachine()
        run = Run("run-state", "task-state")
        machine.transition(run, RunStatus.RUNNING)
        machine.transition(run, RunStatus.PAUSED)
        machine.transition(run, RunStatus.RUNNING)
        machine.transition(run, RunStatus.CANCELLED)
        with self.assertRaisesRegex(ValueError, "非法 Run 状态迁移"):
            machine.transition(run, RunStatus.RUNNING)

    def test_accepted_run_can_fail_before_start(self):
        run = Run("run-init-failure", "task-init-failure")
        RunStateMachine().transition(run, RunStatus.FAILED)
        self.assertEqual(RunStatus.FAILED, run.status)


if __name__ == "__main__":
    unittest.main()
