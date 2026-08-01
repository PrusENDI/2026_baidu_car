# MC602 Arm Response Validation and Reset Settling Design

## Goal

Eliminate two high-risk arm-control behaviors:

1. A delayed or unrelated MC602 response must not be accepted as the acknowledgement for an arm servo command.
2. X/Y reset motion must not begin immediately after the arm command acknowledgement; the arm receives a fixed 1.5 second mechanical settling period.

## Scope

This change covers single-device MC602 transactions through `DevCmdInterface` and the arm reset sequence in `ArmController.reset_position()`.

The following are deliberately out of scope:

- MC602 stream resynchronization and scanning past mismatched frames;
- request sequence identifiers or firmware changes;
- `DevListWrap` batch response validation for chassis encoders;
- changing the 80 ms axis timeout or 200 ms servo timeout;
- changing retry counts or retry delays;
- replacing fixed mechanical settling with position feedback.

## Response Validation

`DevCmdInterface.send_get()` already owns both the request payload and the raw response payload before `get_result()` removes protocol fields. It will validate the response before parsing application data.

For each single-device transaction:

1. Clear `last_data` before sending.
2. Send the request with the existing device timeout.
3. Reject a missing response.
4. Reject a response whose length differs from the request/data structure size.
5. Compare every identity field present in the request structure:
   - `dev_id` at byte 0;
   - operation/action at byte 1 when present;
   - `port_id` at byte 2 when the device structure includes a port.
6. Reject a mismatch without updating `last_data`.
7. Parse the application result only after validation succeeds.
8. Update and return `last_data` only for a validated, non-empty result.

The validator derives the number of identity fields from the request built for that transaction rather than trusting mutable state left by another call. Devices without a port validate only fields that are actually present.

This validation is fail-safe but not a complete request-correlation protocol. Two delayed requests with the same device, action, and port remain indistinguishable without a firmware-provided sequence identifier.

## Arm Command Behavior

`ServoBus_2.set_angle()` continues returning the result of the underlying single-device transaction. `ArmController.set_arm_angle()` keeps its existing maximum of three attempts and 0.5 second retry delay.

- A validated response ends retrying and allows the software `side` and `_arm_angle_last` state to update.
- A missing, malformed, or mismatched response is treated as no acknowledgement and triggers the next attempt.
- Exhausting all attempts returns `False` and leaves the software arm state unchanged.

## Reset Settling Behavior

Define an arm reset settling constant of exactly `1.5` seconds.

`ArmController.reset_position()` performs the following order:

1. Confirm X and Y zero-speed entry state using the existing behavior.
2. Command the hand to `UP`.
3. Send arm `RIGHT` and require a validated acknowledgement.
4. If the arm command fails, raise an error immediately. Do not sleep and do not enter X/Y cooperative reset.
5. If the arm command succeeds, sleep for exactly 1.5 seconds.
6. Only after the sleep completes, enter `_reset_xy_cooperative()`.

The delay represents mechanical settling, not proof that the servo reached its target. It is intentionally fixed rather than configurable for this focused repair.

## Error Handling

- Validation failures return `None` from the MC602 device transaction and never update `last_data`.
- Existing arm retry logging remains responsible for attempt-level diagnostics.
- The validator logs enough expected and actual identity information to distinguish malformed responses from device, action, and port mismatches, without treating the response as successful.
- Reset failures preserve the existing exception behavior and prevent subsequent axis motion.

## Testing

Add isolated tests with a fake MC602 serial transaction boundary. Tests must demonstrate:

1. A matching device/action/port response is accepted.
2. Wrong device, wrong action, wrong port, and wrong-length responses are rejected.
3. Rejected responses leave `last_data` as `None`.
4. `set_arm_angle()` retries after a mismatched response and updates `side` only after a validated response.
5. `reset_position()` calls the 1.5 second settling sleep after a successful arm acknowledgement and before `_reset_xy_cooperative()`.
6. A failed arm command performs neither the settling sleep nor `_reset_xy_cooperative()`.

Tests will be written and observed failing before production code changes, then rerun after the minimal implementation.

## Acceptance Criteria

- An MC602 response from a different device, operation, or port cannot make `set_arm_angle()` succeed.
- A rejected response cannot populate `last_data` or update arm direction state.
- Successful reset always has a 1.5 second interval between validated arm acknowledgement and the first X/Y cooperative reset action.
- Failed arm acknowledgement prevents all X/Y reset motion.
- Targeted tests and the relevant existing test suite pass.
