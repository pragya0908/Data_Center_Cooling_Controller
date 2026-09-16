"""Smoke test: verify temperature rises with min cooling and drops with max cooling."""

from environment.datacenter_env import DataCenterEnv


def run_test(label: str, action: int, steps: int = 50) -> None:
    """Run *steps* iterations with a fixed *action* and print temperatures."""
    env = DataCenterEnv(seed=42)
    print(f"\n{'=' * 60}")
    print(f"  {label}  |  action={action}  |  steps={steps}")
    print(f"{'=' * 60}")
    print(f"{'Step':>6}  {'Temp (°C)':>10}  {'Cooling':>8}  {'Workload':>9}  {'Reward':>8}")
    print(f"{'-' * 60}")

    for step in range(steps):
        state, reward, done = env.step(action)
        if step % 5 == 0 or step == steps - 1:
            print(
                f"{step:>6}  {env.current_temp:>10.2f}  "
                f"{env.current_cooling:>8.2f}  "
                f"{env.current_workload:>9.2f}  "
                f"{reward:>8.2f}"
            )

    return env.current_temp


# --- Test 1: minimum cooling (action 0) – temperature should rise ----------
final_temp_hot = run_test("MIN COOLING (should heat up)", action=0)

# --- Test 2: maximum cooling (action 4) – temperature should drop -----------
final_temp_cold = run_test("MAX COOLING (should cool down)", action=4)

# --- Summary ----------------------------------------------------------------
print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  Final temp (min cooling): {final_temp_hot:.2f} °C")
print(f"  Final temp (max cooling): {final_temp_cold:.2f} °C")

if final_temp_hot > 24.0:
    print("  [PASS] Min cooling -> temperature rose above start (24 C)")
else:
    print("  [FAIL] UNEXPECTED: temperature did not rise with min cooling")

if final_temp_cold < 24.0:
    print("  [PASS] Max cooling -> temperature dropped below start (24 C)")
else:
    print("  [FAIL] UNEXPECTED: temperature did not drop with max cooling")
