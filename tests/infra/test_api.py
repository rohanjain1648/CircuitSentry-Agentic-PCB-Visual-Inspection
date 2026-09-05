import aws_cdk as cdk
from aws_cdk.assertions import Template
from infra.circuitsentry_stack import CircuitSentryStack


def test_api_gateway_has_expected_routes():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::ApiGateway::RestApi", 1)
    methods = template.find_resources("AWS::ApiGateway::Method")
    http_methods = {m["Properties"]["HttpMethod"] for m in methods.values()}
    assert {"POST", "GET"} <= http_methods
