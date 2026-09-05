import json
import re

import aws_cdk as cdk
from aws_cdk.assertions import Template
from infra.circuitsentry_stack import CircuitSentryStack


def test_state_machine_has_choice_state_with_recapture_loop():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::StepFunctions::StateMachine", 1)

    # The synthesized DefinitionString is an Fn::Join of string fragments
    # interleaved with intrinsic references (e.g. the Lambda ARN), so it is
    # not a plain string that Match.string_like_regexp can match directly.
    # Reconstruct it (substituting a placeholder for non-string fragments)
    # and assert that all three terminal branches appear, in order, after
    # the Perception state.
    (state_machine,) = template.find_resources("AWS::StepFunctions::StateMachine").values()
    fragments = state_machine["Properties"]["DefinitionString"]["Fn::Join"][1]
    definition_string = "".join(f if isinstance(f, str) else "<TOKEN>" for f in fragments)

    assert re.search(
        r"Perception.*Recapture.*FlagForApproval.*Pass", definition_string, re.DOTALL
    ), definition_string


def _definition_string(stack) -> str:
    """Reconstruct the Fn::Join'd DefinitionString into a plain string."""
    template = Template.from_stack(stack)
    (state_machine,) = template.find_resources("AWS::StepFunctions::StateMachine").values()
    fragments = state_machine["Properties"]["DefinitionString"]["Fn::Join"][1]
    return "".join(f if isinstance(f, str) else "<TOKEN>" for f in fragments)


def test_perception_task_retries_transient_failures():
    """The client uploads the frame after the execution starts, so the first
    Perception invocation can lose the race and see NoSuchKey."""
    app = cdk.App()
    definition_string = _definition_string(CircuitSentryStack(app, "TestStack"))

    definition = json.loads(definition_string)
    perception = definition["States"]["Perception"]
    # CDK's LambdaInvoke also emits its own default retry for the Lambda
    # service exceptions; ours is the States.ALL catch-all.
    retries = perception["Retry"]
    (retry,) = [r for r in retries if r["ErrorEquals"] == ["States.ALL"]]
    assert retry["IntervalSeconds"] == 2
    assert retry["MaxAttempts"] == 4
    assert retry["BackoffRate"] == 2.0


def test_perception_task_catches_into_manual_review_terminal_state():
    """spec §8: repeated failure must route to manual review, never drop the board."""
    app = cdk.App()
    definition_string = _definition_string(CircuitSentryStack(app, "TestStack"))

    definition = json.loads(definition_string)
    perception = definition["States"]["Perception"]
    (catch,) = perception["Catch"]
    assert catch["ErrorEquals"] == ["States.ALL"]
    assert catch["Next"] == "ManualReview"

    manual_review = definition["States"]["ManualReview"]
    assert manual_review["Type"] == "Pass"
    assert manual_review.get("End") is True  # terminal
