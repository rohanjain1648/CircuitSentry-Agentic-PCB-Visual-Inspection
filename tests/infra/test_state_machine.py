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
